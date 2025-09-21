const std = @import("std");

const mre = @import("mre.zig");
const Lexer = mre.Lexer;

const Tokens = @This();
src: Src,
list: Value.List.Slice,

pub const empty: Tokens = .{
    .src = .empty,
    .list = .empty,
};

/// Only frees `self.src` if `self` was parsed in streaming mode.
pub fn deinit(self: Tokens, gpa: std.mem.Allocator) void {
    self.src.deinit(gpa);
    var list = self.list;
    list.deinit(gpa);
}

pub const Value = extern struct {
    kind: Lexer.Token.Kind,
    loc: Loc,

    pub const Index = u32;
    pub const List = std.MultiArrayList(Value);
    pub const ByteOffset = u32;

    pub const Loc = extern struct {
        start: ByteOffset,
        end: ByteOffset,

        pub fn getSrc(loc: Loc, src: []const u8) []const u8 {
            return src[loc.start..loc.end];
        }
    };
};

pub const Src = union(Lexer.Mode) {
    /// Unowned reference.
    full: []const u8,
    /// Owned memory.
    stream: std.ArrayListUnmanaged(u8),

    pub const empty: Src = .{ .full = "" };

    pub fn deinit(src: Src, gpa: std.mem.Allocator) void {
        switch (src) {
            .full => {},
            .stream => |owned| {
                var mutable = owned;
                mutable.deinit(gpa);
            },
        }
    }

    pub fn slice(src: Src) []const u8 {
        return switch (src) {
            .full => |ref| ref,
            .stream => |owned| owned.items,
        };
    }
};

pub fn tokenizeReader(
    gpa: std.mem.Allocator,
    src: *std.Io.Reader,
) (std.mem.Allocator.Error || Lexer.ReaderError)!Tokens {
    var result: Tokens = .empty;
    errdefer result.deinit(gpa);
    try result.reuseTokenizeReader(gpa, src);
    return result;
}

pub fn reuseTokenizeReader(
    self: *Tokens,
    gpa: std.mem.Allocator,
    src: *std.Io.Reader,
) (std.mem.Allocator.Error || Lexer.ReaderError)!void {
    try self.reuseTokenizeImpl(gpa, src, .stream);
}

/// Asserts `src.len <= maxInt(Value.Index)`.
pub fn tokenizeSlice(
    gpa: std.mem.Allocator,
    src: []const u8,
) std.mem.Allocator.Error!Tokens {
    var result: Tokens = .empty;
    errdefer result.deinit(gpa);
    try result.reuseTokenizeSlice(gpa, src);
    return result;
}

/// Asserts `src.len <= maxInt(Value.Index)`.
pub fn reuseTokenizeSlice(
    self: *Tokens,
    gpa: std.mem.Allocator,
    src: []const u8,
) std.mem.Allocator.Error!void {
    std.debug.assert(src.len <= std.math.maxInt(Value.Index));
    var fixed: std.Io.Reader = .fixed(src);
    self.reuseTokenizeImpl(gpa, &fixed, .full) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        error.ReadFailed => unreachable,
    };
}

fn tokenizeImpl(
    gpa: std.mem.Allocator,
    src: *std.Io.Reader,
    mode: Lexer.Mode,
) (std.mem.Allocator.Error || Lexer.ReaderError)!Tokens {
    var result: Tokens = .empty;
    errdefer result.deinit(gpa);
    try result.reuseTokenizeImpl(gpa, src, mode);
    return result;
}

fn reuseTokenizeImpl(
    self: *Tokens,
    gpa: std.mem.Allocator,
    unlimited_src: *std.Io.Reader,
    mode: Lexer.Mode,
) (std.mem.Allocator.Error || Lexer.ReaderError)!void {
    var limited_buffer: [Lexer.min_stream_buffer_size]u8 = undefined;
    var limited_src = unlimited_src.limited(.limited(std.math.maxInt(Value.Index)), &limited_buffer);
    const src: *std.Io.Reader = switch (mode) {
        .full => unlimited_src,
        .stream => &limited_src.interface,
    };

    var maybe_new_src_buffer: std.ArrayList(u8) = .empty;
    errdefer maybe_new_src_buffer.deinit(gpa);

    const src_buffer: *std.ArrayList(u8) = switch (self.src) {
        .full => &maybe_new_src_buffer,
        .stream => |*owned| owned,
    };
    defer self.src = switch (mode) {
        .full => .{ .full = src.buffer[0..src.end] },
        .stream => .{ .stream = src_buffer.* },
    };
    src_buffer.clearRetainingCapacity();

    var list = self.list.toMultiArrayList();
    defer self.list = list.toOwnedSlice();
    list.clearRetainingCapacity();

    var lexer: Lexer = .init;
    while (true) {
        const tok = try nextToken(gpa, src, mode, &lexer, src_buffer);
        if (list.len == std.math.maxInt(Value.Index)) unreachable;
        try list.append(gpa, tok);
        if (tok.kind == .eof) break;
    }
}

fn nextToken(
    gpa: std.mem.Allocator,
    src: *std.Io.Reader,
    mode: Lexer.Mode,
    lexer: *Lexer,
    src_buffer: *std.ArrayList(u8),
) (std.mem.Allocator.Error || Lexer.ReaderError)!Tokens.Value {
    const first = try lexer.peekToken(src, mode);
    const start: u32 = @intCast(switch (mode) {
        .full => src.seek,
        .stream => src_buffer.items.len,
    });
    var len: u32 = 0;

    len += @intCast(first.getLen());
    switch (mode) {
        .full => {},
        .stream => if (first.sourceLen()) |src_len| {
            try src_buffer.appendSlice(gpa, src.buffered()[0..src_len]);
        },
    }
    src.toss(first.getLen());

    // only attempt to concatenate subsequent tokens if it is concatenatable.
    if (first.sourceLen() != null) while (true) {
        const tok = try lexer.peekToken(src, mode);
        if (tok != first.getKind()) break; // don't toss, we'll need this one later
        switch (mode) {
            .full => {},
            .stream => if (tok.sourceLen()) |src_len| {
                try src_buffer.appendSlice(gpa, src.buffered()[0..src_len]);
            },
        }
        src.toss(tok.getLen());
    };

    return .{
        .kind = first.getKind(),
        .loc = .{
            .start = start,
            .end = start + len,
        },
    };
}
