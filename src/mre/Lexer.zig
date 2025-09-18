const std = @import("std");

const Lexer = @This();
state: State,

pub const init: Lexer = .{
    .state = .start,
};

const State = enum {
    eof,
    start,
    invalid,
    ident,
    number,
};

pub const Token = union(Kind) {
    eof,
    /// Payload is the length of the invalid bytes in `src.buffered()[0..length]`.
    invalid: usize,

    /// Payload is the length of the identifier in `src.buffered()[0..length]`.
    whitespace: usize,
    /// Payload is the length of the identifier in `src.buffered()[0..length]`.
    ident: usize,
    /// Payload is the length of the number in `src.buffered()[0..length]`.
    number: usize,

    let,
    underscore,
    equal,
    colon,
    semicolon,
    comma,

    paren_l,
    paren_r,

    ampersand,
    pipe,
    percent,

    plus,
    plus_pipe,
    plus_percent,

    sub,
    sub_pipe,
    sub_percent,

    pub const Kind = enum(u8) {
        /// Simple empty token placed at the end.
        eof,
        /// Consecutive tokens of this kind are presumed to concatenate to form one token.
        /// Consists of a string of invalid bytes.
        invalid,

        /// Consecutive tokens of this kind are presumed to concatenate to form one token.
        /// Consists of: `(' '|'\t'|'\n'|'\r'|'\v'|'\f')+`.
        whitespace,
        /// Consecutive tokens of this kind are presumed to concatenate to form one token.
        /// Consists of: `[A-Za-z_][A-Za-z_0-9]*`.
        ident,
        /// Consecutive tokens of this kind are presumed to concatenate to form one token.
        /// Consists of: `[0-9][0-9ui_\.]*`.
        number,

        /// Consists of `'let'`.
        let,
        /// Consists of `'_'`.
        underscore,
        /// Consists of `'='`.
        equal,
        /// Consists of `':'`.
        colon,
        /// Consists of `';'`.
        semicolon,
        /// Consists of `','`.
        comma,

        /// Consists of `'('`.
        paren_l,
        /// Consists of `')'`.
        paren_r,

        /// Consists of `'&'`.
        ampersand,
        /// Consists of `'|'`.
        pipe,
        /// Consists of `'%'`.
        percent,

        /// Consists of `'+'`.
        plus,
        /// Consists of `'+|'`.
        plus_pipe,
        /// Consists of `'+%'`.
        plus_percent,

        /// Consists of `'-'`.
        sub,
        /// Consists of `'-|'`.
        sub_pipe,
        /// Consists of `'-%'`.
        sub_percent,

        pub const StaticSrc = union(enum) {
            null,
            eof,
            char: u8,
            str: []const u8,

            pub fn asStrLen(self: *const StaticSrc) ?u8 {
                return if (self.asStr()) |str| @intCast(str.len) else null;
            }

            pub fn asStr(self: *const StaticSrc) ?[]const u8 {
                return switch (self.*) {
                    .null => null,
                    .eof => "",
                    .char => |*byte| byte[0..1],
                    .str => |str| str,
                };
            }
        };

        pub fn staticSrc(self: Kind) StaticSrc {
            return switch (self) {
                .eof => .eof,
                .invalid => .null,
                .whitespace => .null,
                .ident => .null,
                .number => .null,

                .let => .{ .str = "let" },
                .underscore => .{ .char = '_' },
                .equal => .{ .char = '=' },
                .colon => .{ .char = ':' },
                .semicolon => .{ .char = ';' },
                .comma => .{ .char = ',' },
                .paren_l => .{ .char = '(' },
                .paren_r => .{ .char = ')' },

                .ampersand => .{ .char = '&' },
                .pipe => .{ .char = '|' },
                .percent => .{ .char = '%' },

                .plus => .{ .char = '+' },
                .plus_pipe => .{ .str = "+|" },
                .plus_percent => .{ .str = "+%" },

                .sub => .{ .char = '-' },
                .sub_pipe => .{ .str = "-|" },
                .sub_percent => .{ .str = "-%" },
            };
        }
    };

    pub fn getKind(token: Token) Kind {
        return token;
    }

    /// If the token isn't represented by a static string, returns the
    /// length of the string in the source buffer; otherwise returns null.
    pub fn sourceLen(token: Token) ?usize {
        return switch (token) {
            .invalid, .ident, .number, .whitespace => |length| length,
            inline else => |pl| blk: {
                pl; // this should be a discardable void
                break :blk null;
            },
        };
    }

    /// Returns the length of the string that would be returned by `getSrc`.
    pub fn getLen(token: Token) usize {
        return switch (token) {
            inline else => |src_len, kind| switch (@TypeOf(src_len)) {
                usize => src_len,
                void => comptime kind.staticSrc().asStrLen() orelse @compileError(
                    "No static or dynamic string for " ++ @tagName(kind),
                ),
                else => |T| @compileError(
                    "Unhandled: " ++ @typeName(T),
                ),
            },
        };
    }

    pub fn getSrc(
        token: Token,
        /// From `src.buffered()` right after the call to `peekToken` which got this token,
        /// and before the call to `src.toss(token.getLen())`.
        buffered: []const u8,
    ) []const u8 {
        return buffered[0..token.getLen()];
    }
};

pub const Mode = enum {
    /// The given `src` is or acts like `std.Io.Reader.fixed`, providing the full buffer in-memory as `src.buffered()`.
    full,
    /// The given `src` is a general streaming reader, the buffer is only expected to hold a subset of the content at any given time.
    stream,
};

/// In streaming mode the buffer must allow look-ahead of the longest static token src + a delimiter.
pub const min_stream_buffer_size = blk: {
    const longest_kw_len = longest: {
        var longest: usize = 0;
        for (@typeInfo(Token.Kind).@"enum".fields) |field| {
            const kind: Token.Kind = @enumFromInt(field.value);
            longest = @max(longest, kind.staticSrc().asStrLen() orelse 0);
        }
        break :longest longest;
    };
    const min_delimiter_len = 1;
    break :blk longest_kw_len + min_delimiter_len;
};

pub const ReaderError = std.Io.Reader.ShortError;

/// Returns `token`. Call `src.toss(token.getLen())` to advance past the returned token in the reader,
/// so that the subsequent call to `peekToken` will return the next token.
pub fn peekToken(
    lexer: *Lexer,
    src: *std.Io.Reader,
    mode: Mode,
) ReaderError!Token {
    const vt = std.ascii.control_code.vt;
    const ff = std.ascii.control_code.ff;

    switch (mode) {
        .full => {
            // unlike in streaming mode, the buffer size doesn't matter, since
            // we'll never actually end up seeing any more content than there is,
            // so there's no chance that an identifier which starts the same as
            // a keyword could be seen on the edge just before a rebase into the
            // second half of what turns out to be an identifier.
        },
        .stream => {
            std.debug.assert(src.buffer.len >= min_stream_buffer_size);
        },
    }

    var buffered_end: usize = 0;
    const token_kind: Token.Kind, lexer.state = sw: switch (lexer.state) {
        .eof => .{ .eof, .eof },
        .start => {
            std.debug.assert(buffered_end == 0);
            const first_byte = src.peekByte() catch |err| switch (err) {
                error.ReadFailed => |e| return e,
                error.EndOfStream => break :sw .{ .eof, .eof },
            };
            switch (first_byte) {
                'A'...'Z',
                'a'...'z',
                '_',
                => {
                    buffered_end += 1;
                    continue :sw .ident;
                },

                '0'...'9',
                => {
                    buffered_end += 1;
                    continue :sw .number;
                },

                ' ', '\t', '\n', '\r', vt, ff => {
                    buffered_end += 1;
                    src.fill(buffered_end + 1) catch |err| switch (err) {
                        error.ReadFailed => |e| return e,
                        error.EndOfStream => {},
                    };
                    buffered_end = for (
                        src.buffered()[buffered_end..],
                        buffered_end..,
                    ) |c, end| switch (c) {
                        ' ', '\t', '\n', '\r', vt, ff => {},
                        else => break end,
                    } else src.bufferedLen();
                    break :sw .{ .whitespace, .start };
                },

                inline '=', ':', ';', ',', '&', '|', '%', '(', ')' => |char| {
                    const kind: Token.Kind = comptime switch (char) {
                        '=' => .equal,
                        ':' => .colon,
                        ';' => .semicolon,
                        ',' => .comma,
                        '&' => .ampersand,
                        '|' => .pipe,
                        '%' => .percent,
                        '(' => .paren_l,
                        ')' => .paren_r,
                        else => @compileError("Unhandled: '" ++ .{char} ++ "'"),
                    };
                    break :sw .{ kind, .start };
                },

                inline '+', '-' => |char| {
                    const simple, const percent, const pipe = comptime switch (char) {
                        '+' => .{ .plus, .plus_percent, .plus_pipe },
                        '-' => .{ .sub, .sub_percent, .sub_pipe },
                        else => @compileError("Unhandled: '" ++ .{char} ++ "'"),
                    };

                    const eof = try fillCheckEof(src, 2);
                    if (eof) break :sw .{ simple, .start };
                    const kind: Token.Kind = switch (src.buffered()[1]) {
                        '%' => percent,
                        '|' => pipe,
                        else => break :sw .{ simple, .start },
                    };
                    break :sw .{ kind, .start };
                },

                else => continue :sw .invalid,
            }
        },

        .invalid => {
            buffered_end = for (
                src.buffered()[buffered_end..],
                buffered_end..,
            ) |c, end| {
                switch (c) {
                    else => continue,
                    ' ', '\t', '\n', '\r', vt, ff => {},
                    'A'...'Z', 'a'...'z', '_' => {},
                    '0'...'9' => {},
                    ':', ';', '=' => {},
                }
                break end;
            } else src.bufferedLen();
            break :sw .{ .invalid, .start }; // always re-start for invalid tokens
        },

        .ident => {
            std.debug.assert(buffered_end <= 1);
            const eof = try fillCheckEof(src, min_stream_buffer_size);
            buffered_end, const delim: bool = for (
                src.buffered()[buffered_end..],
                buffered_end..,
            ) |char, end| switch (char) {
                'A'...'Z',
                'a'...'z',
                '0'...'9',
                '_',
                => continue,
                else => break .{ end, true },
            } else .{ src.bufferedLen(), false };
            const token_is_done = eof or delim;

            const Kw = enum {
                let,
                @"_",
            };
            if (std.meta.stringToEnum(Kw, src.buffered()[0..buffered_end])) |kw| {
                std.debug.assert(token_is_done);
                const kind: Token.Kind = switch (kw) {
                    .let => .let,
                    ._ => .underscore,
                };
                break :sw .{ kind, .start };
            }
            break :sw .{
                .ident,
                if (token_is_done) .start else .ident,
            };
        },

        .number => {
            std.debug.assert(buffered_end <= 1);
            const eof = try fillCheckEof(src, buffered_end + 1);
            buffered_end, const delim = for (
                src.buffered()[buffered_end..],
                buffered_end..,
            ) |char, end| switch (char) {
                '0'...'9', '_', '.', 'u', 'i' => continue,
                else => break .{ end, true },
            } else .{ src.bufferedLen(), false };
            const token_is_done = eof or delim;
            break :sw .{
                .number,
                if (token_is_done) .start else .number,
            };
        },
    };
    return switch (token_kind) {
        inline else => |kind| @unionInit(
            Token,
            @tagName(kind),
            switch (@FieldType(Token, @tagName(kind))) {
                void => {},
                usize => buffered_end,
                else => |T| @compileError("Unhandled: " ++ @typeName(T)),
            },
        ),
    };
}

/// Wrapper around `src.fill(n)` that returns true for eof instead of an error.
fn fillCheckEof(src: *std.Io.Reader, n: usize) ReaderError!bool {
    src.fill(n) catch |err| switch (err) {
        error.ReadFailed => |e| return e,
        error.EndOfStream => return true,
    };
    return false;
}

const TestToken = struct {
    kind: Token.Kind,
    src: []const u8,

    pub fn init(kind: Token.Kind, src: []const u8) TestToken {
        if (kind.staticSrc().asStr()) |expected_src| {
            if (!std.mem.eql(u8, src, expected_src)) {
                std.debug.panic("{} has static source '{s}', which does not match '{s}'", .{ kind, expected_src, src });
            }
        }
        return .{ .kind = kind, .src = src };
    }

    pub const nl: TestToken = .init(.whitespace, "\n");
    pub const space: TestToken = .init(.whitespace, " ");

    pub fn number(src: []const u8) TestToken {
        return .init(.number, src);
    }

    pub fn ident(src: []const u8) TestToken {
        return .init(.ident, src);
    }

    pub inline fn static(comptime static_kind: Token.Kind) TestToken {
        comptime return .{
            .kind = static_kind,
            .src = static_kind.staticSrc().asStr() orelse @compileError(
                "No static source for '" ++ @tagName(static_kind) ++ "'",
            ),
        };
    }

    pub fn format(self: TestToken, w: *std.Io.Writer) !void {
        try w.print(".{{ {}, '{f}' }}", .{ self.kind, std.zig.fmtString(self.src) });
    }
};

fn lexAndCollapseTestTokens(
    src: *std.Io.Reader,
    mode: Mode,
    gpa: std.mem.Allocator,
    test_tokens: *std.ArrayList(TestToken),
) !void {
    var lexer: Lexer = .init;
    while (true) {
        const tok = try lexer.peekToken(src, mode);
        const tok_src = tok.getSrc(src.buffered());
        defer src.toss(tok.getLen());
        switch (tok) {
            .eof => break,
            inline else => |_, kind| {
                try test_tokens.ensureUnusedCapacity(gpa, 1);

                const new_src: []const u8 = new_src: {
                    const last_tt = test_tokens.getLastOrNull() orelse
                        break :new_src try gpa.dupe(u8, tok_src);
                    if (last_tt.kind != kind)
                        break :new_src try gpa.dupe(u8, tok_src);
                    if (kind.staticSrc() != .null)
                        break :new_src try gpa.dupe(u8, tok_src);

                    _ = test_tokens.pop().?;
                    var last_str_dyn: std.ArrayList(u8) = .fromOwnedSlice(@constCast(last_tt.src));
                    defer last_str_dyn.deinit(gpa);
                    try last_str_dyn.ensureTotalCapacityPrecise(
                        gpa,
                        last_str_dyn.items.len + tok_src.len,
                    );
                    last_str_dyn.appendSliceAssumeCapacity(tok_src);
                    break :new_src try last_str_dyn.toOwnedSlice(gpa);
                };
                test_tokens.appendAssumeCapacity(.{
                    .kind = kind,
                    .src = new_src,
                });
            },
        }
    }
}

fn expectEqlTokens(
    expected: []const TestToken,
    actual: []const TestToken,
) !void {
    const amt = @min(expected.len, actual.len);
    for (expected[0..amt], actual[0..amt], 0..) |expected_tok, actual_tok, index| {
        errdefer std.log.err(
            "Difference occurs at [{d}]: {f} != {f}",
            .{ index, actual_tok, expected_tok },
        );
        try std.testing.expectEqual(expected_tok.kind, actual_tok.kind);
        try std.testing.expectEqualStrings(expected_tok.src, actual_tok.src);
    }

    const TtSliceFmt = struct {
        slice: []const TestToken,

        pub fn format(
            self: @This(),
            w: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            try w.writeAll("{ ");
            for (self.slice, 0..) |elem, i| {
                if (i != 0) try w.writeAll(", ");
                try elem.format(w);
            }
            try w.writeAll(" }");
        }
    };

    switch (std.math.order(expected.len, actual.len)) {
        .lt => {
            const tt_slice_fmt: TtSliceFmt = .{ .slice = actual[expected.len..] };
            std.log.err("Unexpected tokens after index {d}: {f}", .{ amt, tt_slice_fmt });
            return error.TestGotUnexpectedTokens;
        },
        .gt => {
            const tt_slice_fmt: TtSliceFmt = .{ .slice = expected[actual.len..] };
            std.log.err("Missing tokens after index {d}: {f}", .{ amt, tt_slice_fmt });
            return error.TestMissingExpectedTokens;
        },
        .eq => {},
    }
}

fn expectTokenization(
    src: []const u8,
    expected_tokens: []const TestToken,
) !void {
    const gpa = std.testing.allocator;

    if (expected_tokens.len != 0 and expected_tokens[expected_tokens.len - 1].kind == .eof) {
        std.debug.panic("Redundant specification of terminating eof token", .{});
    }

    var fixed_reader: std.Io.Reader = .fixed(src);

    var stream_reader_buffer: [min_stream_buffer_size]u8 = undefined;
    var stream_reader_state: std.testing.Reader = .init(&stream_reader_buffer, &.{
        .{ .buffer = src },
    });
    stream_reader_state.artificial_limit = .limited(1);

    var actual_tokens: std.ArrayList(TestToken) = .empty;
    defer actual_tokens.deinit(gpa);
    defer for (actual_tokens.items) |pair| gpa.free(pair.src);
    try actual_tokens.ensureTotalCapacityPrecise(gpa, expected_tokens.len);

    for ([_]*std.Io.Reader{
        &fixed_reader,
        &stream_reader_state.interface,
    }) |src_reader| {
        for (actual_tokens.items) |pair| gpa.free(pair.src);
        actual_tokens.clearRetainingCapacity();
        try lexAndCollapseTestTokens(src_reader, .full, gpa, &actual_tokens);
        try expectEqlTokens(expected_tokens, actual_tokens.items);
    }
}

test Lexer {
    try expectTokenization("", &.{});
    try expectTokenization(" \n", &.{.init(.whitespace, " \n")});
    try expectTokenization("let", &.{.static(.let)});
    try expectTokenization("foo", &.{.init(.ident, "foo")});
    try expectTokenization("10_024.0", &.{.init(.number, "10_024.0")});
    try expectTokenization("_", &.{.static(.underscore)});
    try expectTokenization(":", &.{.static(.colon)});
    try expectTokenization("=", &.{.static(.equal)});
    try expectTokenization(";", &.{.static(.semicolon)});
    try expectTokenization("(", &.{.static(.paren_l)});
    try expectTokenization(")", &.{.static(.paren_r)});

    try expectTokenization("&", &.{.static(.ampersand)});
    try expectTokenization("|", &.{.static(.pipe)});
    try expectTokenization("%", &.{.static(.percent)});

    try expectTokenization("+", &.{.static(.plus)});
    try expectTokenization("+%", &.{.static(.plus_percent)});
    try expectTokenization("+|", &.{.static(.plus_pipe)});

    try expectTokenization("-", &.{.static(.sub)});
    try expectTokenization("-%", &.{.static(.sub_percent)});
    try expectTokenization("-|", &.{.static(.sub_pipe)});

    try expectTokenization("3u8 +% 255u8", &.{
        .number("3u8"),
        .space,
        .static(.plus_percent),
        .space,
        .number("255u8"),
    });

    try expectTokenization("(a:u8) - 255u8", &.{
        .static(.paren_l),
        .ident("a"),
        .static(.colon),
        .ident("u8"),
        .static(.paren_r),
        .space,
        .static(.sub),
        .space,
        .number("255u8"),
    });

    try expectTokenization(
        \\let foo: u32 = 10_024.0:u8;
        \\_ = foo;
        \\
    ,
        &.{
            .static(.let),
            .space,
            .ident("foo"),
            .static(.colon),
            .space,
            .ident("u32"),
            .space,
            .static(.equal),
            .space,
            .number("10_024.0"),
            .static(.colon),
            .ident("u8"),
            .static(.semicolon),
            .nl,

            .static(.underscore),
            .space,
            .static(.equal),
            .space,
            .ident("foo"),
            .static(.semicolon),
            .nl,
        },
    );
}
