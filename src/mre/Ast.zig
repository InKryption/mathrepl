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
    un_op: UnOp,
    bin_op: BinOp,
    list: List,
    block: Block,

    pub const null_init: Node = .{ .null = .{ 0, .init(0, 0) } };

    pub const Tag = @typeInfo(Node).@"union".tag_type.?;
    pub const Index = enum(Int) {
        pub const Int = u32;
        null = 0,
        _,

        pub const root: Index = .null;

        pub fn fromInt(index: Int) Index {
            return @enumFromInt(index);
        }

        pub fn toInt(self: Index) ?Int {
            return switch (self) {
                .null => null,
                _ => |value| @intFromEnum(value),
            };
        }

        pub fn toIntAllowRoot(self: Index) Int {
            return @intFromEnum(self);
        }
    };

    pub const ValueRef = struct {
        main_token: Tokens.Value.Index,
        unused_data: Packed.Data = .init(0, 0),

        pub const Kind = enum { ident, number };

        pub fn getKind(ref: ValueRef, tokens: Tokens) Kind {
            const tok = ref.getRawToken(tokens);
            return switch (tok.kind) {
                .ident => .ident,
                .number => .number,
                else => unreachable, // bad value ref
            };
        }

        pub fn getSrc(ref: ValueRef, tokens: Tokens) []const u8 {
            const tok = ref.getRawToken(tokens);
            return tok.loc.getSrc(tokens.src.slice());
        }

        pub fn getRawToken(ref: ValueRef, tokens: Tokens) Tokens.Value {
            return tokens.list.get(ref.main_token);
        }

        pub fn unpack(main_token: Tokens.Value.Index, data: Packed.Data) ValueRef {
            return .{
                .main_token = main_token,
                .unused_data = data,
            };
        }

        pub fn pack(ref: ValueRef) struct { Tokens.Value.Index, Packed.Data } {
            return .{ ref.main_token, ref.unused_data };
        }
    };

    pub const UnOp = struct {
        op: Tokens.Value.Index,
        operand: Node.Index,
        unused_rhs: u32 = 0,

        pub fn unpack(main_token: Tokens.Value.Index, data: Packed.Data) UnOp {
            return .{
                .op = main_token,
                .operand = .fromInt(data.lhs),
                .unused_rhs = data.rhs,
            };
        }

        pub fn pack(un_op: UnOp) struct { Tokens.Value.Index, Packed.Data } {
            return .{ un_op.op, .init(un_op.operand.toInt().?, un_op.unused_rhs) };
        }
    };

    pub const BinOp = struct {
        lhs: Node.Index,
        op: Tokens.Value.Index,
        rhs: Node.Index,

        pub fn unpack(main_token: Tokens.Value.Index, data: Packed.Data) BinOp {
            return .{
                .lhs = .fromInt(data.lhs),
                .op = main_token,
                .rhs = .fromInt(data.rhs),
            };
        }

        pub fn pack(bin_op: BinOp) struct { Tokens.Value.Index, Packed.Data } {
            return .{ bin_op.op, .init(bin_op.lhs.toInt().?, bin_op.rhs.toInt().?) };
        }
    };

    pub const Grouped = struct {
        expr: Node.Index,
        paren_l: Tokens.Value.Index,
        paren_r: Tokens.Value.Index,

        pub fn unpack(main_token: Tokens.Value.Index, data: Packed.Data) Grouped {
            return .{
                .expr = .fromInt(data.lhs),
                .paren_l = main_token,
                .paren_r = data.rhs,
            };
        }

        pub fn pack(grouped: Grouped) struct { Tokens.Value.Index, Packed.Data } {
            return .{ grouped.paren_l, .init(grouped.expr.toInt().?, grouped.paren_r) };
        }
    };

    pub const List = struct {
        start: u32,
        end: u32,
        /// Either points to the trailing comma, or to the last token of the last element in the list.
        end_tok: Tokens.Value.Index,

        pub fn getNodesLen(list: List) u32 {
            return list.end - list.start;
        }

        pub fn getNodes(
            list: List,
            /// The `extra_data` field of `Ast`.
            extra_data: []const u32,
        ) []const Node.Index {
            return @ptrCast(extra_data[list.start..list.end]);
        }

        pub fn unpack(main_token: Tokens.Value.Index, data: Packed.Data) List {
            return .{
                .end_tok = main_token,
                .start = data.lhs,
                .end = data.rhs,
            };
        }

        pub fn pack(list: List) struct { Tokens.Value.Index, Packed.Data } {
            return .{ list.end_tok, .init(list.start, list.end) };
        }
    };

    pub const Block = struct {
        start: u32,
        end: u32,
        /// NOTE: for the root block, this is eof.
        closing_brace: Tokens.Value.Index,

        pub fn getNodesLen(block: Block) u32 {
            return block.end - block.start;
        }

        pub fn getNodes(
            block: Block,
            /// The `extra_data` field of `Ast`.
            extra_data: []const u32,
        ) []const Node.Index {
            return @ptrCast(extra_data[block.start..block.end]);
        }

        pub fn unpack(main_token: Tokens.Value.Index, data: Packed.Data) Block {
            return .{
                .closing_brace = main_token,
                .start = data.lhs,
                .end = data.rhs,
            };
        }

        pub fn pack(block: Block) struct { Tokens.Value.Index, Packed.Data } {
            return .{ block.closing_brace, .init(block.start, block.end) };
        }
    };

    pub const Packed = extern struct {
        tag: Tag,
        main_token: Tokens.Value.Index,
        data: Data,

        pub const Data = extern struct {
            lhs: u32,
            rhs: u32,

            pub fn init(lhs: u32, rhs: u32) Data {
                return .{ .lhs = lhs, .rhs = rhs };
            }
        };

        pub const List = std.MultiArrayList(Node.Packed);

        pub fn unpack(self: Packed) Node {
            return switch (self.tag) {
                inline else => |tag| @unionInit(Node, @tagName(tag), .unpack(self.main_token, self.data)),
            };
        }

        pub fn pack(node: Node) Packed {
            const main_token: Tokens.Value.Index, const data: Data = switch (node) {
                inline else => |pl| pl.pack(),
            };
            return .{
                .tag = node,
                .main_token = main_token,
                .data = data,
            };
        }

        comptime {
            const e_fields = @typeInfo(Tag).@"enum".fields;
            @setEvalBranchQuota(e_fields.len * 3 + 1);
            for (e_fields) |e_field| {
                const packed_value: Node.Packed = .{
                    .tag = @enumFromInt(e_field.value),
                    .main_token = 1,
                    .data = .init(2, 3),
                };
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
    params: NodeFmt.Params,
) NodeFmt {
    return .{
        .ast = ast,
        .tokens = tokens,
        .params = params,
    };
}

pub const NodeFmt = struct {
    ast: Ast,
    tokens: Tokens,
    params: Params,

    pub const Params = struct {
        node: Node.Index,
        /// Buffer for walking the tree of nodes.
        /// The number of elements directly corresponds to the maximum nesting level,
        /// after which point any deeper nodes in the AST will be truncated.
        /// To avoid any truncation, should be `walk_buffer.len == ast.nodes.len`.
        walk_buffer: []u64,
        options: Options,

        pub const Options = struct {
            precedence_delims: ?[2][]const u8,
            truncated_str: []const u8,

            pub const default: Options = .{
                .precedence_delims = null,
                .truncated_str = "<...>",
            };

            pub const default_prec_delim: Options = .{
                .precedence_delims = .{ "(", ")" },
                .truncated_str = "<...>",
            };
        };
    };

    pub fn format(
        self: NodeFmt,
        w: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        const params = self.params;
        const options = params.options;
        const four_spaces = " " ** 4;

        const Pending = packed struct(u64) {
            node: Node.Index,
            need_prec_paren: bool,
            state: packed union {
                const Int = u31;
                const initial: Int = 0;
                raw: enum(Int) {
                    initial = initial,
                    _,
                },
                grouped: enum(Int) {
                    start = initial,
                    end,
                    _,
                },
                bin_op: enum(Int) {
                    start = initial,
                    op,
                    close_precedence_delim,
                    _,
                },
                block: packed struct(Int) {
                    index: Int = initial,
                },
            },
        };
        var pending_stack: std.ArrayList(Pending) = .initBuffer(@ptrCast(params.walk_buffer));
        pending_stack.appendBounded(.{
            .node = params.node,
            .need_prec_paren = false,
            .state = .{ .raw = .initial },
        }) catch {
            try w.writeAll(options.truncated_str);
            return;
        };
        var indent: u32 = 0;
        while (true) {
            const pending: Pending = pending_stack.pop() orelse break;
            const pending_node = self.ast.nodes.get(pending.node.toIntAllowRoot());
            switch (pending_node.unpack()) {
                .value_ref => |value_ref| {
                    std.debug.assert(pending.state.raw == .initial);
                    const src = value_ref.getSrc(self.tokens);
                    try w.print("{f}", .{std.zig.fmtString(src)});
                },
                .grouped => |grouped| switch (pending.state.grouped) {
                    .start => {
                        pending_stack.appendSliceBounded(&.{
                            .{
                                .node = pending.node,
                                .need_prec_paren = false,
                                .state = .{ .grouped = .end },
                            },
                            .{
                                .node = grouped.expr,
                                .need_prec_paren = false,
                                .state = .{ .raw = .initial },
                            },
                        }) catch {
                            try writeVecAllCopy(w, .{ "(", options.truncated_str, ")" });
                            continue;
                        };
                        try w.writeByte('(');
                    },
                    .end => {
                        try w.writeByte(')');
                    },
                    _ => unreachable,
                },
                .un_op => |un| {
                    std.debug.assert(pending.state.raw == .initial);
                    const op_tok = self.tokens.list.get(un.op);
                    const op_str = op_tok.loc.getSrc(self.tokens.src.slice());
                    try w.writeAll(op_str);
                    pending_stack.appendAssumeCapacity(.{
                        .node = un.operand,
                        .need_prec_paren = false,
                        .state = .{ .raw = .initial },
                    });
                },
                .bin_op => |bin_op| {
                    const op_tok = self.tokens.list.get(bin_op.op);
                    const op_str = op_tok.loc.getSrc(self.tokens.src.slice());
                    const canon_spacing = oper_table.get(op_tok.kind.toOperator().?).canon_spacing;
                    const before: []const u8, const after: []const u8 = switch (canon_spacing) {
                        .surround => .{ " ", " " },
                        .stick_to_left => .{ "", " " },
                    };
                    bin_op_sw: switch (pending.state.bin_op) {
                        .start => {
                            if (pending.need_prec_paren) {
                                if (options.precedence_delims) |delims| {
                                    const start_delim, _ = delims;
                                    try w.writeAll(start_delim);
                                }
                            }
                            pending_stack.appendSliceBounded(&.{
                                .{
                                    .node = pending.node,
                                    .need_prec_paren = pending.need_prec_paren,
                                    .state = .{ .bin_op = .op },
                                },
                                .{
                                    .node = bin_op.lhs,
                                    .need_prec_paren = true,
                                    .state = .{ .raw = .initial },
                                },
                            }) catch {
                                try writeVecAllCopy(w, .{ options.truncated_str, before, op_str, after, options.truncated_str });
                                continue :bin_op_sw .close_precedence_delim;
                            };
                        },
                        .op => {
                            try writeVecAllCopy(w, .{ before, op_str, after });
                            // if we got here, it means that the `appendSliceBounded` call in the `start` branch
                            // was successful in appending 2 nodes, and that capacity shouldn't be any different
                            // at this point.
                            pending_stack.appendSliceAssumeCapacity(&.{
                                .{
                                    .node = pending.node,
                                    .need_prec_paren = pending.need_prec_paren,
                                    .state = .{ .bin_op = .close_precedence_delim },
                                },
                                .{
                                    .node = bin_op.rhs,
                                    .need_prec_paren = true,
                                    .state = .{ .raw = .initial },
                                },
                            });
                        },
                        .close_precedence_delim => {
                            if (pending.need_prec_paren) {
                                if (options.precedence_delims) |delims| {
                                    _, const end_delim = delims;
                                    try w.writeAll(end_delim);
                                }
                            }
                        },
                        _ => unreachable,
                    }
                },
                .list => |list| std.debug.panic("TODO: {}", .{list}),
                .block => |block| {
                    var state = pending.state.block;
                    const block_nodes = block.getNodes(self.ast.extra_data);
                    if (block_nodes.len == 0) {
                        std.debug.assert(state.index == 0);
                        try w.writeAll("{}");
                        continue;
                    }

                    const target_node_index = state.index;
                    if (target_node_index == block_nodes.len) {
                        try w.writeByte(';');
                        if (block_nodes.len == 1) {
                            try w.writeByte(' ');
                        } else {
                            indent -= 1;
                            try w.writeByte('\n');
                            try w.splatBytesAll(four_spaces, indent);
                        }
                        try w.writeByte('}');
                        continue;
                    }

                    const target_node = block_nodes[target_node_index];
                    state.index += 1;

                    pending_stack.appendSliceBounded(&.{
                        .{
                            .node = pending.node,
                            .need_prec_paren = false,
                            .state = .{ .block = state },
                        },
                        .{
                            .node = target_node,
                            .need_prec_paren = false,
                            .state = .{ .raw = .initial },
                        },
                    }) catch {
                        std.debug.assert(target_node_index == 0); // this has to be the first time we're actually here
                        try writeVecAllCopy(w, .{ "{ ", options.truncated_str, " }" });
                        continue;
                    };

                    if (target_node_index == 0) {
                        try w.writeByte('{');
                        if (block_nodes.len == 1) {
                            try w.writeByte(' ');
                        } else {
                            try w.writeByte('\n');
                            indent += 1;
                        }
                    } else {
                        try w.writeByte(';');
                        if (block_nodes.len != 1) {
                            try w.writeByte('\n');
                        }
                    }

                    try w.splatBytesAll(four_spaces, indent);
                },
            }
        }
    }
};

fn writeVecAllCopy(
    w: *std.Io.Writer,
    vec: anytype,
) std.Io.Writer.Error!void {
    var copy: [vec.len][]const u8 = vec;
    try w.writeVecAll(&copy);
}

const OperInfo = struct {
    prec: u32,
    assoc: Assoc,
    canon_spacing: CanonSpacing = .surround,

    const Assoc = enum { none, left, right };
    const CanonSpacing = enum { surround, stick_to_left };

    fn init(
        prec: u32,
        assoc: Assoc,
        canon_spacing: CanonSpacing,
    ) OperInfo {
        return .{
            .prec = prec,
            .assoc = assoc,
            .canon_spacing = canon_spacing,
        };
    }

    fn choose(lhs: OperInfo, rhs: OperInfo) enum { invalid, lhs, rhs } {
        return switch (std.math.order(lhs.prec, rhs.prec)) {
            .eq => if (lhs.assoc != rhs.assoc)
                .invalid
            else switch (lhs.assoc) {
                .none => .invalid,
                .left => .lhs,
                .right => .rhs,
            },
            .lt => .rhs,
            .gt => .lhs,
        };
    }
};

const oper_table: std.EnumArray(Lexer.Token.Kind.Operator, OperInfo) = .init(.{
    .colon = .init(3, .left, .stick_to_left),
    .ampersand = .init(1, .left, .surround),
    .pipe = .init(1, .left, .surround),
    .percent = .init(2, .left, .surround),
    .slash = .init(2, .left, .surround),

    .add = .init(1, .left, .surround),
    .add_wrap = .init(1, .left, .surround),
    .add_saturate = .init(1, .left, .surround),

    .sub = .init(1, .left, .surround),
    .sub_wrap = .init(1, .left, .surround),
    .sub_saturate = .init(1, .left, .surround),

    .mul = .init(2, .left, .surround),
    .mul_wrap = .init(2, .left, .surround),
    .mul_saturate = .init(2, .left, .surround),
});

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

    var states: std.ArrayList(Parser.State) = .empty;
    defer states.deinit(gpa);

    var parser: Parser = .{
        .tokens = &tokens,
        .tokens_index = 0,
        .nodes = &nodes,
        .extra_data = &extra_data,
        .scratch = &scratch,
        .states = &states,
    };
    try parser.parse(gpa);

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
    states: *std.ArrayList(State),

    const ParseError = error{ParseFail};

    fn parse(parser: *Parser, gpa: std.mem.Allocator) !void {
        const tokens_kind: []const Lexer.Token.Kind = parser.tokens.list.items(.kind);

        const root_node = try parser.addNode(gpa, undefined);
        std.debug.assert(root_node == Node.Index.root);

        try parser.states.append(gpa, .{ .expect_block_statements = .{
            .dst_node = root_node,
            .scratch_start = @intCast(parser.scratch.items.len),
        } });
        mainloop: while (parser.states.pop()) |state| switch (state) {
            .expect_block_statements => |data| {
                parser.skipWhitespace();
                const scratch_start: u32 = data.scratch_start;
                std.debug.assert(parser.scratch.items.len >= scratch_start);
                defer std.debug.assert(parser.scratch.items.len >= scratch_start);

                const closing_brace: Tokens.Value.Index = switch (tokens_kind[parser.tokens_index]) {
                    .eof, .brace_r => parser.tokens_index,
                    else => {
                        const block_item_node = try parser.addNode(gpa, undefined);
                        try parser.scratch.append(gpa, block_item_node.toInt().?);
                        try parser.states.appendSlice(gpa, &.{
                            .{ .expect_block_statements = data },
                            .{
                                .expect_statement_or_expr = .{
                                    .dst_node = block_item_node,
                                },
                            },
                        });
                        continue :mainloop;
                    },
                };

                const extra_start: u32 = @intCast(parser.extra_data.items.len);
                try parser.extra_data.appendSlice(gpa, parser.scratch.items[scratch_start..]);
                const extra_end: u32 = @intCast(parser.extra_data.items.len);
                parser.scratch.shrinkRetainingCapacity(scratch_start);

                parser.nodes.set(data.dst_node.toIntAllowRoot(), .pack(.{ .block = .{
                    .start = extra_start,
                    .end = extra_end,
                    .closing_brace = closing_brace,
                } }));
            },
            .expect_closing_brace => switch (tokens_kind[parser.tokens_index]) {
                .brace_r => parser.tokens_index += 1,
                else => std.debug.panic("TODO: handle missing closing brace", .{}),
            },
            .expect_statement_or_expr => |data| switch (tokens_kind[parser.tokens_index]) {
                else => |t| std.debug.panic("TODO: {t}", .{t}),
                .whitespace => unreachable,
                .ident,
                .number,
                .paren_l,
                .brace_l,
                .sub,
                .sub_wrap,
                => {
                    try parser.states.appendSlice(gpa, &[_]State{
                        .handle_semicolon_after_expr,
                    } ++ State.expectFullExpr(data.dst_node));
                },
            },
            .expect_expr_primary => |data| {
                sw: switch (tokens_kind[parser.tokens_index]) {
                    else => unreachable,
                    .whitespace => {
                        parser.skipWhitespace();
                        continue :sw tokens_kind[parser.tokens_index];
                    },
                    .sub, .sub_wrap => {
                        const op_tok = parser.tokens_index;
                        parser.tokens_index += 1;

                        if (tokens_kind[parser.tokens_index] == .whitespace) {
                            std.debug.panic("TODO: whitespace in between unary operator and operand.", .{});
                        }

                        const operand_node = try parser.addNode(gpa, undefined);
                        parser.nodes.set(data.dst_node.toInt().?, .pack(.{ .un_op = .{
                            .op = op_tok,
                            .operand = operand_node,
                        } }));
                        parser.states.appendAssumeCapacity(.{
                            .expect_expr_primary = .{
                                .dst_node = operand_node,
                            },
                        });
                    },
                    .ident, .number => {
                        parser.nodes.set(data.dst_node.toInt().?, .pack(.{
                            .value_ref = parser.consumeValueRef(),
                        }));
                    },
                    .paren_l => {
                        parser.states.appendAssumeCapacity(.{
                            .expect_grouped_start = .{
                                .dst_node = data.dst_node,
                            },
                        });
                    },
                    .brace_l => {
                        parser.tokens_index += 1;
                        try parser.states.appendSlice(gpa, &.{
                            .expect_closing_brace,
                            .{
                                .expect_block_statements = .{
                                    .dst_node = data.dst_node,
                                    .scratch_start = @intCast(parser.scratch.items.len),
                                },
                            },
                        });
                    },
                }
            },
            .handle_expr_secondary => |data| {
                parser.skipWhitespace();
                switch (tokens_kind[parser.tokens_index]) {
                    else => |tag| std.debug.panic("TODO: {t}", .{tag}),
                    .whitespace => unreachable,
                    .eof, .semicolon, .paren_r, .brace_r => {
                        // do nothing, let the next popped state handle it now that we're done
                    },

                    .ampersand,
                    .pipe,
                    .percent,
                    .slash,

                    .add,
                    .add_wrap,
                    .add_saturate,

                    .sub,
                    .sub_wrap,
                    .sub_saturate,

                    .mul,
                    .mul_wrap,
                    .mul_saturate,

                    .colon,
                    => {
                        const rhs_op_tok = parser.tokens_index;
                        parser.tokens_index += 1;
                        parser.skipWhitespace();
                        const rhs_expr_node = try parser.addNode(gpa, undefined);
                        try parser.states.appendSlice(gpa, &.{
                            .{
                                .join_expr_secondary = .{
                                    .dst_node = data.dst_node,
                                    .rhs_op_tok = rhs_op_tok,
                                    .rhs_expr = rhs_expr_node,
                                },
                            },
                            .{
                                .expect_expr_primary = .{
                                    .dst_node = rhs_expr_node,
                                },
                            },
                        });
                    },
                }
            },
            .join_expr_secondary => |data| {
                parser.skipWhitespace();

                try parser.states.append(gpa, .{
                    // re-queue until we run into a terminating token
                    .handle_expr_secondary = .{
                        .dst_node = data.dst_node,
                    },
                });

                const rhs_op_tok = data.rhs_op_tok;
                const rhs_op_tag = tokens_kind[rhs_op_tok];
                const rhs_op_info = oper_table.get(rhs_op_tag.toOperator().?);

                const current_expr = parser.nodes.get(data.dst_node.toInt().?);
                switch (current_expr.unpack()) {
                    .value_ref, .grouped, .un_op, .list, .block => {
                        const lhs_expr_node = try parser.addNode(gpa, current_expr);
                        parser.nodes.set(data.dst_node.toInt().?, .pack(.{ .bin_op = .{
                            .lhs = lhs_expr_node,
                            .op = rhs_op_tok,
                            .rhs = data.rhs_expr,
                        } }));
                        continue :mainloop;
                    },
                    .bin_op => {},
                }

                const BindWhich = enum { lhs, rhs };

                var lhs_bin_op_index: Node.Index = data.dst_node;
                const bind_which: BindWhich = while (true) {
                    const lhs_bin_op = parser.nodes.get(lhs_bin_op_index.toInt().?).unpack().bin_op;
                    const lhs_op_tok = lhs_bin_op.op;
                    const lhs_op_tag = tokens_kind[lhs_op_tok];
                    const lhs_op_info = oper_table.get(lhs_op_tag.toOperator().?);
                    const bind_which: BindWhich = switch (OperInfo.choose(lhs_op_info, rhs_op_info)) {
                        .invalid => std.debug.panic("TODO: Handle invalid associativity", .{}),
                        .lhs => .lhs,
                        .rhs => .rhs,
                    };
                    switch (bind_which) {
                        .lhs => break .lhs,
                        .rhs => switch (parser.nodes.items(.tag)[lhs_bin_op.rhs.toInt().?]) {
                            .value_ref, .grouped, .un_op, .list, .block => break .rhs,
                            .bin_op => {
                                lhs_bin_op_index = lhs_bin_op.rhs;
                                continue;
                            },
                        },
                    }
                };

                const lhs_bin_op = parser.nodes.get(lhs_bin_op_index.toInt().?).unpack().bin_op;

                switch (bind_which) {
                    .lhs => {
                        const lhs_expr_node = try parser.addNode(gpa, .pack(.{ .bin_op = lhs_bin_op }));
                        const new_bin_op: Node.BinOp = .{
                            .lhs = lhs_expr_node,
                            .op = rhs_op_tok,
                            .rhs = data.rhs_expr,
                        };
                        parser.nodes.set(lhs_bin_op_index.toInt().?, .pack(.{ .bin_op = new_bin_op }));
                    },
                    .rhs => {
                        switch (parser.nodes.items(.tag)[lhs_bin_op.rhs.toInt().?]) {
                            .bin_op => unreachable,
                            .value_ref, .grouped, .un_op, .list, .block => {},
                        }
                        const rhs_outer_expr_node = try parser.addNode(gpa, .pack(.{ .bin_op = .{
                            .lhs = lhs_bin_op.rhs,
                            .op = rhs_op_tok,
                            .rhs = data.rhs_expr,
                        } }));
                        parser.nodes.items(.data)[lhs_bin_op_index.toInt().?].rhs = rhs_outer_expr_node.toInt().?;
                    },
                }
            },
            .expect_grouped_start => |data| {
                std.debug.assert(tokens_kind[parser.tokens_index] == .paren_l);

                const paren_l = parser.tokens_index;
                parser.tokens_index += 1;
                parser.skipWhitespace();

                const inner_expr_node = try parser.addNode(gpa, undefined);
                try parser.states.appendSlice(gpa, &[_]State{
                    .{
                        .expect_grouped_end = .{
                            .dst_node = data.dst_node,
                            .paren_l = paren_l,
                            .inner_expr = inner_expr_node,
                        },
                    },
                } ++ State.expectFullExpr(inner_expr_node));
            },
            .expect_grouped_end => |data| switch (tokens_kind[parser.tokens_index]) {
                else => unreachable,
                .eof, .brace_r, .semicolon => std.debug.panic("TODO: handle unclosed paren", .{}),
                .paren_r => {
                    const paren_r = parser.tokens_index;
                    parser.tokens_index += 1;
                    parser.nodes.set(data.dst_node.toInt().?, .pack(.{ .grouped = .{
                        .expr = data.inner_expr,
                        .paren_l = data.paren_l,
                        .paren_r = paren_r,
                    } }));
                },
            },
            .handle_semicolon_after_expr => {
                if (tokens_kind[parser.tokens_index] == .semicolon) {
                    parser.tokens_index += 1;
                }
            },
        };

        const block = parser.nodes.get(root_node.toIntAllowRoot()).unpack().block;
        if (parser.tokens.list.get(block.closing_brace).kind != .eof) {
            std.debug.panic("TODO: handle trailing rbrace", .{});
        }
    }

    const State = union(enum) {
        expect_block_statements: struct {
            dst_node: Node.Index,
            scratch_start: u32,
        },
        expect_closing_brace,
        expect_statement_or_expr: struct {
            dst_node: Node.Index,
        },
        expect_expr_primary: struct {
            dst_node: Node.Index,
        },
        handle_expr_secondary: struct {
            dst_node: Node.Index,
        },
        join_expr_secondary: struct {
            dst_node: Node.Index,
            rhs_op_tok: Tokens.Value.Index,
            rhs_expr: Node.Index,
        },
        expect_grouped_start: struct {
            dst_node: Node.Index,
        },
        expect_grouped_end: struct {
            dst_node: Node.Index,
            paren_l: Tokens.Value.Index,
            inner_expr: Node.Index,
        },
        handle_semicolon_after_expr,

        fn expectFullExpr(dst_node: Node.Index) [2]State {
            return .{
                .{
                    .handle_expr_secondary = .{
                        .dst_node = dst_node,
                    },
                },
                .{
                    .expect_expr_primary = .{
                        .dst_node = dst_node,
                    },
                },
            };
        }
    };

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
        node: Node.Packed,
    ) std.mem.Allocator.Error!Node.Index {
        try self.nodes.ensureUnusedCapacity(gpa, 1);
        return self.addNodeAssumeCapacity(node);
    }

    fn addNodeAssumeCapacity(
        self: *const Parser,
        node: Node.Packed,
    ) Node.Index {
        const index = self.nodes.addOneAssumeCapacity();
        self.nodes.set(index, node);
        return .fromInt(@intCast(index));
    }

    fn consumeValueRef(self: *Parser) Node.ValueRef {
        const tokens_kind: []const Lexer.Token.Kind = self.tokens.list.items(.kind);
        switch (tokens_kind[self.tokens_index]) {
            .ident, .number => {},
            else => unreachable,
        }
        defer self.tokens_index += 1;
        return .{
            .main_token = self.tokens_index,
        };
    }
};

const TestNode = union(Node.Tag) {
    value_ref: struct { Node.ValueRef.Kind, []const u8 },
    grouped: *const TestNode,
    un_op: struct { Lexer.Token.Kind, *const TestNode },
    bin_op: struct {
        lhs: *const TestNode,
        op: Lexer.Token.Kind,
        rhs: *const TestNode,
    },
    list: []const TestNode,
    block: []const TestNode,

    fn initValueRef(kind: Node.ValueRef.Kind, str: []const u8) TestNode {
        return .{ .value_ref = .{ kind, str } };
    }

    fn initUnOp(op: Lexer.Token.Kind, operand: *const TestNode) TestNode {
        return .{ .un_op = .{ op, operand } };
    }

    fn initBinOp(
        op: Lexer.Token.Kind,
        params: struct {
            lhs: *const TestNode,
            rhs: *const TestNode,
        },
    ) TestNode {
        return .{ .bin_op = .{
            .lhs = params.lhs,
            .op = op,
            .rhs = params.rhs,
        } };
    }

    fn initList(nodes: []const TestNode) TestNode {
        return .{ .list = nodes };
    }

    fn initBlock(nodes: []const TestNode) TestNode {
        return .{ .block = nodes };
    }
};

fn expectEqualAstNode(
    ast: Ast,
    tokens: Tokens,
    actual_base_node: Node.Index,
    expected: TestNode,
) !void {
    const gpa = std.testing.allocator;

    const State = union(enum) {
        const State = @This();
        any: struct {
            expected: TestNode,
            actual_node: Node.Index,
        },
        cmp_slices: struct {
            expected: []const TestNode,
            actual: []const Node.Index,
            index: usize,
        },
    };
    var states: std.ArrayList(State) = .empty;
    defer states.deinit(gpa);
    try states.append(gpa, .{ .any = .{
        .expected = expected,
        .actual_node = actual_base_node,
    } });
    while (states.pop()) |state| switch (state) {
        .any => |any| {
            const actual_packed = ast.nodes.get(any.actual_node.toIntAllowRoot());
            try std.testing.expectEqual(@as(Node.Tag, any.expected), actual_packed.tag);
            switch (any.expected) {
                .value_ref => |expected_valref| {
                    const expected_kind, const expected_str = expected_valref;
                    const actual_valref = actual_packed.unpack().value_ref;
                    try std.testing.expectEqualStrings(expected_str, actual_valref.getSrc(tokens));
                    try std.testing.expectEqual(expected_kind, actual_valref.getKind(tokens));
                },
                .grouped => |expected_grouped| {
                    const actual_grouped = actual_packed.unpack().grouped;
                    states.appendAssumeCapacity(.{ .any = .{
                        .expected = expected_grouped.*,
                        .actual_node = actual_grouped.expr,
                    } });
                },
                .un_op => |expected_un_op| {
                    const expected_kind, const expected_operand = expected_un_op;
                    const actual_un_op = actual_packed.unpack().un_op;
                    try std.testing.expectEqual(expected_kind, tokens.list.get(actual_un_op.op).kind);
                    states.appendAssumeCapacity(.{ .any = .{
                        .expected = expected_operand.*,
                        .actual_node = actual_un_op.operand,
                    } });
                },
                .bin_op => |expected_bin_op| {
                    const actual_bin_op = actual_packed.unpack().bin_op;
                    try std.testing.expectEqual(expected_bin_op.op, tokens.list.get(actual_bin_op.op).kind);
                    try states.appendSlice(gpa, &.{
                        .{ .any = .{
                            .expected = expected_bin_op.rhs.*,
                            .actual_node = actual_bin_op.rhs,
                        } },
                        .{ .any = .{
                            .expected = expected_bin_op.lhs.*,
                            .actual_node = actual_bin_op.lhs,
                        } },
                    });
                },
                .list => |expected_list| {
                    const actual_list = actual_packed.unpack().list;
                    states.appendAssumeCapacity(.{ .cmp_slices = .{
                        .expected = expected_list,
                        .actual = actual_list.getNodes(ast.extra_data),
                        .index = 0,
                    } });
                },
                .block => |expected_block| {
                    const actual_block = actual_packed.unpack().block;
                    states.appendAssumeCapacity(.{ .cmp_slices = .{
                        .expected = expected_block,
                        .actual = actual_block.getNodes(ast.extra_data),
                        .index = 0,
                    } });
                },
            }
        },
        .cmp_slices => |slices| cmp_slices: {
            const amt = @min(slices.actual.len, slices.expected.len);
            if (slices.index == amt) {
                try std.testing.expectEqual(slices.expected.len, slices.actual.len);
                break :cmp_slices;
            }
            const expected_item = slices.expected[slices.index];
            const actual_item = slices.actual[slices.index];

            try states.appendSlice(gpa, &.{
                .{ .cmp_slices = updated: {
                    var updated = slices;
                    updated.index += 1;
                    break :updated updated;
                } },
                .{ .any = .{
                    .expected = expected_item,
                    .actual_node = actual_item,
                } },
            });
        },
    };
}

test Ast {
    const gpa = std.testing.allocator;

    const tokens: Tokens = try .tokenizeSlice(gpa,
        \\{
        \\    (32u8 + 1) * 3:u8;
        \\}
    );
    defer tokens.deinit(gpa);

    const ast: Ast = try .parse(gpa, tokens);
    defer ast.deinit(gpa);

    try expectEqualAstNode(ast, tokens, .root, .initBlock(&.{
        .initBlock(&.{
            .initBinOp(.mul, .{
                .lhs = &.{ .grouped = &.initBinOp(.add, .{
                    .lhs = &.initValueRef(.number, "32u8"),
                    .rhs = &.initValueRef(.number, "1"),
                }) },
                .rhs = &.initBinOp(.colon, .{
                    .lhs = &.initValueRef(.number, "3"),
                    .rhs = &.initValueRef(.ident, "u8"),
                }),
            }),
        }),
    }));
}
