const std = @import("std");

const mre = @import("mre.zig");
const Lexer = mre.Lexer;
const Tokens = mre.Tokens;

const Ast = @This();
nodes: Node.Packed.List.Slice,
extra_data: []const u32,

pub fn deinit(ast: Ast, gpa: std.mem.Allocator) void {
    var nodes = ast.nodes;
    nodes.deinit(gpa);
    gpa.free(ast.extra_data);
}

pub const Node = union(enum(u8)) {
    value_ref: ValueRef,
    grouped: Grouped,
    bin_op: BinOp,
    block: Block,

    pub const Tag = @typeInfo(Node).@"union".tag_type.?;
    pub const Index = u32;
    pub const root_index: Index = 0;
    pub const null_index: Index = root_index;

    pub const ValueRef = struct {
        main_token: Tokens.Value.Index,

        pub fn getSrc(ref: ValueRef, tokens: Tokens) []const u8 {
            const tok = tokens.list.get(ref.main_token);
            return tokens.src.slice()[tok.start..tok.end];
        }
    };

    pub const BinOp = struct {
        lhs: Node.Index,
        op: Tokens.Value.Index,
        rhs: Node.Index,
    };

    pub const Grouped = struct {
        expr: Node.Index,
        paren_l: Tokens.Value.Index,
        paren_r: Tokens.Value.Index,
    };

    pub const Block = struct {
        start: u32,
        end: u32,

        pub fn getNodesLen(block: Block) u32 {
            return block.end - block.start;
        }

        pub fn getNodes(block: Block, ast: Ast) []const Node.Index {
            return ast.extra_data[block.start..block.end];
        }
    };

    pub const Packed = extern struct {
        tag: Tag,
        main_token: Tokens.Value.Index,
        data: Data,

        pub const Data = extern struct {
            lhs: u32,
            rhs: u32,
        };

        pub const List = std.MultiArrayList(Node.Packed);

        pub fn unpack(self: Packed) Node {
            return switch (self.tag) {
                .value_ref => .{ .value_ref = .{
                    .main_token = self.main_token,
                } },
                .bin_op => .{ .bin_op = .{
                    .lhs = self.data.lhs,
                    .op = self.main_token,
                    .rhs = self.data.rhs,
                } },
                .grouped => .{ .grouped = .{
                    .expr = self.data.lhs,
                    .paren_l = self.main_token,
                    .paren_r = self.data.rhs,
                } },
                .block => .{ .block = .{
                    .start = self.data.lhs,
                    .end = self.data.rhs,
                } },
            };
        }

        pub fn pack(self: Node) Packed {
            const main_token: Tokens.Value.Index, //
            const data: Data //
            = switch (self) {
                .value_ref => |int| .{
                    int.main_token,
                    .{ .lhs = 0, .rhs = 0 },
                },
                .bin_op => |bin_op| .{
                    bin_op.op,
                    .{ .lhs = bin_op.lhs, .rhs = bin_op.rhs },
                },
                .grouped => |grouped| .{
                    grouped.paren_l,
                    .{ .lhs = grouped.expr, .rhs = grouped.paren_r },
                },
                .block => |block| .{
                    0,
                    .{ .lhs = block.start, .rhs = block.end },
                },
            };
            return .{
                .tag = self,
                .main_token = main_token,
                .data = data,
            };
        }

        comptime {
            const e_fields = @typeInfo(Tag).@"enum".fields;
            @setEvalBranchQuota(e_fields.len * 3 + 1);
            for (e_fields) |e_field| {
                const tag: Tag = @enumFromInt(e_field.value);
                const original: Node = @unionInit(Node, e_field.name, switch (tag) {
                    .value_ref => .{ .main_token = 123 },
                    .bin_op => .{
                        .lhs = 6,
                        .op = 7,
                        .rhs = 8,
                    },
                    .grouped => .{
                        .expr = 2,
                        .paren_l = 20,
                        .paren_r = 200,
                    },
                    .block => .{
                        .start = 10,
                        .end = 30,
                    },
                });
                const packed_value: Node.Packed = .pack(original);
                const unpacked_value: Node = packed_value.unpack();
                const repacked_value: Packed = .pack(unpacked_value);
                if (repacked_value.tag == packed_value.tag and
                    repacked_value.main_token == packed_value.main_token and
                    repacked_value.data.lhs == packed_value.data.lhs and
                    repacked_value.data.rhs == packed_value.data.rhs)
                {
                    continue;
                }
                @compileError(
                    std.fmt.comptimePrint(
                        \\During `pack(unpack(expected))`,
                        \\expected = {}
                        \\actual   = {}
                    ,
                        .{ packed_value, repacked_value },
                    ),
                );
            }
        }
    };
};

pub fn nodeFmt(
    ast: Ast,
    tokens: Tokens,
    params: struct {
        node: Node.Index,
        /// Buffer for walking the tree of nodes.
        walk_buffer: []u64,
    },
) NodeFmt {
    return .{
        .node = params.node,
        .ast = ast,
        .tokens = tokens,
        .walk_buffer = params.walk_buffer,
    };
}

pub const NodeFmt = struct {
    node: Node.Index,
    ast: Ast,
    tokens: Tokens,
    /// Buffer for walking the tree of nodes.
    walk_buffer: []u64,

    pub fn format(
        self: NodeFmt,
        w: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        const truncated_str = "<...>";

        const Pending = packed struct(u64) {
            kind: enum(u2) {
                token,
                /// Ignores the `index` and `data` fields.
                node_block,
                node,
            },
            index: u32,
            data: packed union {
                token: enum(u30) { generic, space_surround, _ },
                node_block: packed struct(u30) {
                    index: u30,
                },
                node: enum(u30) { empty, _ },
            },
        };
        var pending_stack: std.ArrayList(Pending) = .initBuffer(@ptrCast(self.walk_buffer));
        var first_pop: ?Pending = .{
            .kind = .node,
            .index = self.node,
            .data = .{ .node = .empty },
        };
        var indent: u30 = 0;
        while (pending_stack.pop() orelse first_pop) |pending| {
            first_pop = null;
            switch (pending.kind) {
                .token => {
                    const tok = self.tokens.list.get(pending.index);
                    const tok_src = tok.loc.getSrc(self.tokens.src.slice());
                    switch (pending.data.token) {
                        .generic => try w.writeAll(tok_src),
                        .space_surround => {
                            var surrounded = [_][]const u8{ " ", tok_src, " " };
                            try w.writeVecAll(&surrounded);
                        },
                        _ => unreachable,
                    }
                },
                .node_block => {
                    const nodes = self.ast.nodes.get(pending.index).unpack().block.getNodes(self.ast);
                    std.debug.assert(nodes.len != 0);
                    if (pending.data.node_block.index != 0) {
                        try w.writeByte(';');
                    }
                    if (pending.data.node_block.index == nodes.len) {
                        if (nodes.len > 1) {
                            indent -= 1;
                            try w.writeByte('\n');
                            try w.splatBytesAll(" " ** 4, indent);
                        } else {
                            try w.writeByte(' ');
                        }
                        try w.writeByte('}');
                    } else {
                        pending_stack.appendSliceAssumeCapacity(&.{ // checked when we appended this node_block the first time
                            .{
                                .kind = .node_block,
                                .index = pending.index,
                                .data = .{ .node_block = .{ .index = pending.data.node_block.index + 1 } },
                            },
                            .{
                                .kind = .node,
                                .index = nodes[pending.data.node_block.index],
                                .data = .{ .node = .empty },
                            },
                        });
                        if (nodes.len > 1) {
                            try w.writeByte('\n');
                            try w.splatBytesAll(" " ** 4, indent);
                        }
                    }
                },
                .node => switch (self.ast.nodes.get(pending.index).unpack()) {
                    .value_ref => |value_ref| {
                        const tok = self.tokens.list.get(value_ref.main_token);
                        const tok_src = tok.loc.getSrc(self.tokens.src.slice());
                        try w.print("{s}", .{tok_src});
                    },
                    .grouped => |grouped| {
                        pending_stack.appendSliceBounded(&.{
                            .{
                                .kind = .token,
                                .index = grouped.paren_r,
                                .data = .{ .token = .generic },
                            },
                            .{
                                .kind = .node,
                                .index = grouped.expr,
                                .data = .{ .token = .generic },
                            },
                        }) catch {
                            try w.writeAll("(" ++ truncated_str ++ ")");
                            continue;
                        };
                        try w.writeByte('(');
                    },
                    .bin_op => |bin_op| {
                        pending_stack.appendSliceBounded(&.{
                            .{
                                .kind = .node,
                                .index = bin_op.rhs,
                                .data = .{ .node = .empty },
                            },
                            .{
                                .kind = .token,
                                .index = bin_op.op,
                                .data = .{ .token = .space_surround },
                            },
                            .{
                                .kind = .node,
                                .index = bin_op.lhs,
                                .data = .{ .node = .empty },
                            },
                        }) catch {
                            const op_static_str = self.tokens.list.get(bin_op.op).kind.staticSrc();
                            const op_str = op_static_str.asStr().?;
                            var truncated_bin_op_strs = [_][]const u8{ truncated_str ++ " ", op_str, " " ++ truncated_str };
                            try w.writeVecAll(&truncated_bin_op_strs);
                        };
                    },
                    .block => |block| {
                        // see the `node_block` branch.
                        if (pending_stack.unusedCapacitySlice().len < 2) {
                            try w.writeAll("{" ++ truncated_str ++ "}");
                            continue;
                        }
                        const nodes = block.getNodes(self.ast);
                        pending_stack.appendBounded(.{
                            .kind = .node_block,
                            .index = pending.index,
                            .data = .{ .node_block = .{ .index = 0 } },
                        }) catch {
                            try w.writeAll("{" ++ truncated_str ++ "}");
                            continue;
                        };
                        try w.writeByte('{');
                        if (nodes.len > 1) {
                            indent += 1;
                        } else {
                            try w.writeByte(' ');
                        }
                    },
                },
            }
        }
    }
};

pub fn parse(
    gpa: std.mem.Allocator,
    tokens: Tokens,
) (std.mem.Allocator.Error || Parser.ParseError)!Ast {
    var nodes: Node.Packed.List = .empty;
    defer nodes.deinit(gpa);

    var extra_data: std.ArrayList(u32) = .empty;
    defer extra_data.deinit(gpa);

    var scratch: std.ArrayList(u32) = .empty;
    defer scratch.deinit(gpa);

    var parser: Parser = .{
        .tokens = &tokens,
        .tokens_index = 0,
        .nodes = &nodes,
        .extra_data = &extra_data,
        .scratch = &scratch,
    };
    const root = try parser.addNode(gpa, .{ .block = undefined });
    while (try parser.expectStatementOrExpr(gpa)) |statement_or_expr| {
        try scratch.append(gpa, statement_or_expr);
    }
    const root_block_start = extra_data.items.len;
    try extra_data.appendSlice(gpa, scratch.items);
    const root_block_end = extra_data.items.len;
    nodes.set(root, .pack(.{ .block = .{
        .start = @intCast(root_block_start),
        .end = @intCast(root_block_end),
    } }));

    var nodes_final = nodes.toOwnedSlice();
    errdefer nodes_final.deinit(gpa);

    const extra_data_final = try extra_data.toOwnedSlice(gpa);
    errdefer gpa.free(extra_data_final);

    return .{
        .nodes = nodes_final,
        .extra_data = extra_data_final,
    };
}

const Parser = struct {
    tokens: *const Tokens,
    tokens_index: Tokens.Value.Index,
    nodes: *Node.Packed.List,
    extra_data: *std.ArrayList(u32),
    scratch: *std.ArrayList(u32),

    const ParseError = error{ParseFail};

    /// Assumes `self.tokens_index < self.tokens.list.len`.
    fn skipWhitespace(self: *Parser) void {
        const tokens_kind: []const Lexer.Token.Kind = self.tokens.list.items(.kind);
        if (tokens_kind[self.tokens_index] == .eof) return;
        while (tokens_kind[self.tokens_index] == .whitespace) {
            self.tokens_index += 1;
        }
    }

    fn addNode(
        self: *const Parser,
        gpa: std.mem.Allocator,
        node: Node,
    ) std.mem.Allocator.Error!Node.Index {
        try self.nodes.ensureUnusedCapacity(gpa, 1);
        return self.addNodeAssumeCapacity(node);
    }

    fn addNodeAssumeCapacity(
        self: *const Parser,
        node: Node,
    ) Node.Index {
        const index = self.nodes.addOneAssumeCapacity();
        self.nodes.set(index, .pack(node));
        return @intCast(index);
    }

    /// Same as `eatAndAddValueRefAssumeCapacity` but allocates automatically.
    fn eatAndAddValueRef(
        self: *Parser,
        gpa: std.mem.Allocator,
    ) std.mem.Allocator.Error!Node.Index {
        try self.nodes.ensureUnusedCapacity(gpa, 1);
        return self.eatAndAddValueRefAssumeCapacity();
    }

    /// On success, increments `self.tokens_index` by one, after returning a value ref node index pointing to the current token.
    /// Asserts the current token is either of kind `ident` or `number`.
    fn eatAndAddValueRefAssumeCapacity(
        self: *Parser,
    ) Node.Index {
        const tokens_kind: []const Lexer.Token.Kind = self.tokens.list.items(.kind);
        switch (tokens_kind[self.tokens_index]) {
            .ident, .number => {},
            else => unreachable,
        }
        const value_ref_node = self.addNodeAssumeCapacity(.{
            .value_ref = .{
                .main_token = self.tokens_index,
            },
        });
        self.tokens_index += 1;
        return value_ref_node;
    }

    pub fn expectStatementOrExpr(
        self: *Parser,
        gpa: std.mem.Allocator,
    ) (ParseError || std.mem.Allocator.Error)!?Node.Index {
        const tokens_kind: []const Lexer.Token.Kind = self.tokens.list.items(.kind);

        self.skipWhitespace();
        switch (tokens_kind[self.tokens_index]) {
            else => |t| std.debug.panic("TODO: {t}", .{t}),
            .whitespace => unreachable,

            .eof => return null,
            .semicolon => {
                std.debug.panic("TODO: gracefully report unneeded semicolon.", .{});
            },
            .ident, .number, .paren_l => {
                return try self.expectExpr(gpa);
            },
        }
    }

    const OperInfo = struct {
        prec: u32,
        assoc: enum { none, left, right },
    };
    const oper_table: std.EnumMap(Lexer.Token.Kind, OperInfo) = .init(.{
        .ampersand = .{ .prec = 1, .assoc = .left },
        .pipe = .{ .prec = 1, .assoc = .left },
        .percent = .{ .prec = 2, .assoc = .left },
        .slash = .{ .prec = 2, .assoc = .left },

        .plus = .{ .prec = 1, .assoc = .left },
        .plus_pipe = .{ .prec = 1, .assoc = .left },
        .plus_percent = .{ .prec = 1, .assoc = .left },

        .sub = .{ .prec = 1, .assoc = .left },
        .sub_pipe = .{ .prec = 1, .assoc = .left },
        .sub_percent = .{ .prec = 1, .assoc = .left },

        .mul = .{ .prec = 2, .assoc = .left },
        .mul_pipe = .{ .prec = 2, .assoc = .left },
        .mul_percent = .{ .prec = 2, .assoc = .left },
    });

    fn expectExpr(
        self: *Parser,
        gpa: std.mem.Allocator,
    ) (ParseError || std.mem.Allocator.Error)!?Node.Index {
        const tokens_kind: []const Lexer.Token.Kind = self.tokens.list.items(.kind);

        const scratch_start = self.scratch.items.len;
        defer self.scratch.shrinkRetainingCapacity(scratch_start);
        defer std.debug.assert(self.scratch.items.len >= scratch_start);
        try self.scratch.ensureUnusedCapacity(gpa, 1);

        self.skipWhitespace();
        while (true) switch (tokens_kind[self.tokens_index]) {
            else => |tag| std.debug.panic("TODO: {t}", .{tag}),
            .whitespace => unreachable,
            .eof => break,
            .semicolon => {
                self.tokens_index += 1;
                break;
            },

            .paren_l => std.debug.panic("TODO", .{}),
            .ident, .number => append_ref: {
                try self.scratch.ensureUnusedCapacity(gpa, 1);
                const ref_node = try self.eatAndAddValueRef(gpa);
                self.skipWhitespace();

                if (self.scratch.items.len == scratch_start) {
                    self.scratch.appendAssumeCapacity(ref_node);
                    break :append_ref;
                }
                const prev_node: Node.Index = self.scratch.pop().?;
                switch (self.nodes.get(prev_node).unpack()) {
                    else => |lhs_node_unpacked| std.debug.panic("TODO: {}", .{lhs_node_unpacked}),
                    .bin_op => {
                        const innermost = getInnerMostBinOpRhs(self.nodes.slice(), prev_node).?;
                        if (self.nodes.get(innermost).unpack().bin_op.rhs != Node.null_index) {
                            std.debug.panic("TODO: gracefully report bad syntax", .{});
                        }
                        self.nodes.items(.data)[innermost].rhs = ref_node;
                        self.scratch.appendAssumeCapacity(prev_node);
                    },
                }
            },

            .ampersand,
            .pipe,
            .percent,
            .slash,

            .plus,
            .plus_pipe,
            .plus_percent,

            .sub,
            .sub_pipe,
            .sub_percent,

            .mul,
            .mul_pipe,
            .mul_percent,
            => |op_tag| {
                const op_tok = self.tokens_index;
                const op_info = oper_table.get(op_tag).?;
                self.tokens_index += 1;
                self.skipWhitespace();

                try self.nodes.ensureUnusedCapacity(gpa, 1);

                if (self.scratch.items.len == scratch_start) {
                    std.debug.panic("TODO: binary op without lhs operand", .{});
                }
                const lhs_node: Node.Index = self.scratch.pop().?;

                switch (self.nodes.get(lhs_node).unpack()) {
                    else => |lhs_node_unpacked| std.debug.panic("TODO: {}", .{lhs_node_unpacked}),
                    .value_ref => {
                        self.scratch.appendAssumeCapacity(self.addNodeAssumeCapacity(.{ .bin_op = .{
                            .lhs = lhs_node,
                            .op = op_tok,
                            .rhs = Node.null_index,
                        } }));
                    },
                    .bin_op => |lhs_bin_op| {
                        const lhs_op_tok = lhs_bin_op.op;
                        const lhs_op_tag = tokens_kind[lhs_op_tok];
                        const lhs_op_info = oper_table.get(lhs_op_tag).?;

                        if (lhs_bin_op.rhs == Node.null_index) {
                            std.debug.panic("TODO: gracefully report chained binary ops without operand in the middle syntax error", .{});
                        }

                        const bind_which: enum { left, right } = switch (std.math.order(lhs_op_info.prec, op_info.prec)) {
                            .eq => blk: {
                                if (lhs_op_info.assoc != op_info.assoc) {
                                    std.debug.panic("TODO: incompatible associativity", .{});
                                }
                                break :blk switch (lhs_op_info.assoc) {
                                    .none => std.debug.panic("TODO: disallowed associativity", .{}),
                                    .left => .left,
                                    .right => .right,
                                };
                            },
                            .lt => .right,
                            .gt => .left,
                        };

                        switch (bind_which) {
                            .left => {
                                self.scratch.appendAssumeCapacity(self.addNodeAssumeCapacity(.{ .bin_op = .{
                                    .lhs = lhs_node,
                                    .op = op_tok,
                                    .rhs = Node.null_index,
                                } }));
                            },
                            .right => {
                                self.nodes.items(.data)[lhs_node].rhs = self.addNodeAssumeCapacity(.{ .bin_op = .{
                                    .lhs = lhs_bin_op.rhs,
                                    .op = op_tok,
                                    .rhs = Node.null_index,
                                } });
                            },
                        }
                    },
                }
            },
        };

        if (self.scratch.items.len == scratch_start) return null;
        std.debug.assert(self.scratch.items.len == scratch_start + 1);
        return self.scratch.getLastOrNull().?;
    }
};

/// Returns null if `outermost` isn't a `bin_op`.
/// Otherwise, returns the node index for the rightmost nested bin_op in the binop tree.
fn getInnerMostBinOpRhs(
    nodes: Node.Packed.List.Slice,
    outermost: Node.Index,
) ?Node.Index {
    const nodes_tag: []const Node.Tag = nodes.items(.tag);
    if (nodes_tag[outermost] != .bin_op) return null;

    var current: Node.Index = outermost;
    while (true) {
        const data = nodes.get(current).unpack().bin_op;
        if (data.rhs == Node.null_index) break;
        if (nodes_tag[data.rhs] != .bin_op) break;
        current = data.rhs;
    }

    return current;
}

test Ast {
    const gpa = std.testing.allocator;

    const tokens: Tokens = try .tokenizeSlice(gpa,
        \\32u8 + 1
    );
    defer tokens.deinit(gpa);

    const ast: Ast = try .parse(gpa, tokens);
    defer ast.deinit(gpa);
}
