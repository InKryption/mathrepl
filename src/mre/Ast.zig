const std = @import("std");

const mre = @import("mre.zig");
const Lexer = mre.Lexer;
const Tokens = mre.Tokens;

const Ast = @This();
root: Node.Index,
nodes: Node.Packed.List.Slice,
extra_data: []const u32,

pub fn deinit(ast: Ast, gpa: std.mem.Allocator) void {
    var nodes = ast.nodes;
    nodes.deinit(gpa);
    gpa.free(ast.extra_data);
}

pub const Node = union(enum(u8)) {
    value_ref: struct {
        main_token: Tokens.Value.Index,
    },
    bin_op: struct {
        lhs: Node.Index,
        op: Tokens.Value.Index,
        rhs: Node.Index,
    },

    pub const Tag = @typeInfo(Node).@"union".tag_type.?;
    pub const Index = u32;
    pub const Packed = extern struct {
        main_token: Tokens.Value.Index,
        tag: Tag,
        data: Data,

        pub const Data = extern struct {
            lhs: u32,
            rhs: u32,
        };

        pub const List = std.MultiArrayList(Node.Packed);

        pub fn unpack(self: Packed) Node {
            return switch (self.tag) {
                .int => .{
                    .main_token = self.main_token,
                },
            };
        }

        pub fn pack(self: Node) Packed {
            return switch (self) {
                .value_ref => |int| .{
                    .main_token = int.main_token,
                    .tag = self,
                    .data = .{ .lhs = 0, .rhs = 0 },
                },
            };
        }
    };
};

pub fn parse(
    gpa: std.mem.Allocator,
    tokens: Tokens,
) (std.mem.Allocator.Error || Parser.Error)!Ast {
    var nodes: Node.Packed.List = .empty;
    defer nodes.deinit(gpa);

    var extra_data: std.ArrayList(u32) = .empty;
    defer extra_data.deinit(gpa);

    var parser: Parser = .{
        .tokens = &tokens,
        .tokens_index = 0,
        .nodes = &nodes,
        .extra_data = &extra_data,
    };
    const root = try parser.expectStatement(gpa, &tokens) orelse 0;

    var nodes_final = nodes.toOwnedSlice();
    errdefer nodes_final.deinit(gpa);

    const extra_data_final = try extra_data.toOwnedSlice(gpa);
    errdefer gpa.free(extra_data_final);

    return .{
        .root = root,
        .nodes = nodes_final,
        .extra_data = extra_data_final,
    };
}

const Parser = struct {
    tokens: *const Tokens,
    tokens_index: Tokens.Value.Index,
    nodes: *Node.Packed.List,
    extra_data: *std.ArrayList(u32),

    const Error = error{ParseFail};

    fn nextToken(self: *Parser) ?Tokens.Value {
        if (self.tokens_index == self.tokens.list.len) return null;
        defer self.tokens_index += 1;
        return self.tokens.list.get(self.tokens_index);
    }

    fn nextTokenSkipWhitespace(self: *Parser) ?Tokens.Value.Index {
        while (true) {
            const tok_idx = self.tokens_index;
            const tok = self.nextToken() orelse return null;
            if (tok.kind == .whitespace) continue;
            return tok_idx;
        }
    }

    fn addNode(
        self: *const Parser,
        gpa: std.mem.Allocator,
        node: Node,
    ) std.mem.Allocator.Error!Node.Index {
        const index = try self.nodes.addOne(gpa);
        self.nodes.set(index, node);
        return @intCast(index);
    }

    fn expectStatement(
        self: *Parser,
        gpa: std.mem.Allocator,
    ) (std.mem.Allocator.Error || Error)!?Node.Index {
        const first_tok_index: Tokens.Value.Index =
            self.nextTokenSkipWhitespace() orelse return null;
        const first_tok: Tokens.Value = self.tokens.list.get(first_tok_index);

        switch (first_tok.kind) {
            else => std.debug.panic("TODO", .{}),
            .number, .ident => {
                const first_node: Node.Index = try self.addNode(gpa, .{
                    .value_ref = .{ .main_token = first_tok_index },
                });
                const second_tok_index: Tokens.Value.Index =
                    self.nextTokenSkipWhitespace() orelse return first_node;
                const second_tok: Tokens.Value = self.tokens.list.get(second_tok_index);
                switch (second_tok.kind) {
                    else => std.debug.panic("TODO", .{}),
                    .semicolon => return first_node,

                    .ampersand,
                    .pipe,
                    .percent,

                    .plus,
                    .plus_pipe,
                    .plus_percent,

                    .sub,
                    .sub_pipe,
                    .sub_percent,
                    => {},
                }
            },
        }
    }
};

test Ast {
    const gpa = std.testing.allocator;

    const tokens: Tokens = try .tokenizeSlice(gpa,
        \\32u8 + 1
    );
    defer tokens.deinit(gpa);

    const ast: Ast = try .parse(gpa, tokens);
    defer ast.deinit(gpa);
}
