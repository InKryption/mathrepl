const std = @import("std");
const big_int_math = std.math.big.int;

const builtin = @import("builtin");

const mre = @import("mre.zig");
const Lexer = mre.Lexer;
const Tokens = mre.Tokens;
const Ast = mre.Ast;

const BigIntLimb = std.math.big.Limb;

const Ir = @This();
insts: Inst.Packed.List.Slice,
root_end: u32,
extra_data: []const u32,
big_int_limbs: []const BigIntLimb,

pub fn deinit(ir: Ir, gpa: std.mem.Allocator) void {
    var insts = ir.insts;
    insts.deinit(gpa);
    gpa.free(ir.extra_data);
    gpa.free(ir.big_int_limbs);
}

pub const Inst = union(Tag) {
    int_pos: u64,
    int_neg: i64,
    int_big: IntBig,
    float: Float128,

    typed: Typed,

    eq: BinOp,
    lt: BinOp,
    lt_eq: BinOp,
    gt: BinOp,
    gt_eq: BinOp,

    negate: UnOp,
    negate_wrap: UnOp,

    div: BinOp,

    add: BinOp,
    add_wrap: BinOp,
    add_saturate: BinOp,

    sub: BinOp,
    sub_wrap: BinOp,
    sub_saturate: BinOp,

    mul: BinOp,
    mul_wrap: BinOp,
    mul_saturate: BinOp,

    if_true: IfTrue,
    if_else: IfElse,

    block: Block,

    pub const Index = enum(Int) {
        pub const Int = u32;
        null = std.math.maxInt(Int),
        _,

        /// `maxInt(Int)` is interpreted as `.null`.
        pub fn fromInt(int: Int) Index {
            return @enumFromInt(int);
        }

        pub fn toInt(self: Index) ?Int {
            return switch (self) {
                .null => null,
                _ => |value| @intFromEnum(value),
            };
        }
    };

    pub fn pack(self: Inst) Packed {
        return .pack(self);
    }

    pub fn unpack(packed_inst: Packed) Inst {
        return packed_inst.unpack();
    }

    pub const Tag = @Type(.{ .@"enum" = .{
        .tag_type = u8,
        .is_exhaustive = true,
        .fields = fields: {
            const u_fields = @typeInfo(Packed.Data).@"union".fields;
            var e_fields: [u_fields.len]std.builtin.Type.EnumField = undefined;
            for (&e_fields, u_fields, 0..) |*e_field, u_field, i| {
                e_field.* = .{ .name = u_field.name, .value = i };
            }
            break :fields &e_fields;
        },
        .decls = &.{},
    } });

    pub const Packed = extern struct {
        tag: Tag,
        data: Data,

        pub const List = std.MultiArrayList(Packed);

        comptime {
            if (@sizeOf(Data) != 8) @compileError(
                std.fmt.comptimePrint("Wrong size '{d}'", .{@sizeOf(Data)}),
            );
        }
        pub const Data = extern union {
            int_pos: u64,
            int_neg: i64,
            int_big: IntBig,
            float: Float128,

            typed: Typed,

            eq: BinOp,

            lt: BinOp,
            lt_eq: BinOp,

            gt: BinOp,
            gt_eq: BinOp,

            negate: UnOp,
            negate_wrap: UnOp,

            div: BinOp,

            add: BinOp,
            add_wrap: BinOp,
            add_saturate: BinOp,

            sub: BinOp,
            sub_wrap: BinOp,
            sub_saturate: BinOp,

            mul: BinOp,
            mul_wrap: BinOp,
            mul_saturate: BinOp,

            if_true: IfTrue,
            if_else: IfElse,

            block: Block,
        };

        pub fn unpack(self: Packed) Inst {
            return switch (self.tag) {
                inline else => |tag| @unionInit(
                    Inst,
                    @tagName(tag),
                    @field(self.data, @tagName(tag)),
                ),
            };
        }

        pub fn pack(inst: Inst) Packed {
            return .{
                .tag = inst,
                .data = switch (inst) {
                    inline else => |pl, tag| @unionInit(Data, @tagName(tag), pl),
                },
            };
        }
    };

    pub const IntBig = packed struct(u64) {
        start: u32,
        len: u31,
        positive: bool,

        pub fn getBigIntConst(
            int_big: IntBig,
            /// The `big_int_limbs` field of `Ir`.
            big_int_limbs: []const BigIntLimb,
        ) big_int_math.Const {
            return .{
                .limbs = big_int_limbs[int_big.start..][0..big_int_limbs.len],
                .positive = int_big.positive,
            };
        }
    };

    pub const Float128 = packed struct(u64) {
        start: u64,

        pub fn get(
            float128: Float128,
            /// The `extra_data` field of `Ir`.
            extra_data: []const u32,
        ) f128 {
            const le_int = extraDataBitCast(u128, extra_data[float128.start..][0 .. @sizeOf(u128) / @sizeOf(u32)]).*;
            return @bitCast(std.mem.littleToNative(u128, le_int));
        }
    };

    pub const Typed = extern struct {
        operand: Inst.Index,
        type: Inst.Index,
    };

    pub const UnOp = packed struct(u32) {
        operand: Inst.Index,
    };

    pub const BinOp = extern struct {
        lhs: Inst.Index,
        rhs: Inst.Index,
    };

    pub const IfTrue = extern struct {
        cond: Inst.Index,
        true_branch: Inst.Index,
    };

    pub const IfElse = extern struct {
        cond: Inst.Index,
        branches: u32,

        pub fn getBranches(
            if_else: IfElse,
            /// The `extra_data` field of `Ir`.
            extra_data: []const u32,
        ) *const [2]Inst.Index {
            return @ptrCast(extra_data[if_else.branches..][0..2]);
        }
    };

    pub const Block = extern struct {
        start: u32,
        end: u32,

        pub fn getInsts(
            block: Block,
            /// The `extra_data` field of `Ir`.
            extra_data: []const u32,
        ) []const Inst.Index {
            return @ptrCast(extra_data[block.start..block.end]);
        }
    };
};

fn extraDataBitCast(
    comptime T: type,
    extra_data: *const [@divExact(@sizeOf(T), @sizeOf(u32))]u32,
) *align(@alignOf(u32)) const T {
    return @ptrCast(extra_data[0 .. @sizeOf(T) / @sizeOf(u32)]);
}

pub fn generate(
    gpa: std.mem.Allocator,
    tokens: Tokens,
    ast: Ast,
) std.mem.Allocator.Error!Ir {
    var insts: Inst.Packed.List = .empty;
    defer insts.deinit(gpa);

    var extra_data: std.ArrayList(u32) = .empty;
    defer extra_data.deinit(gpa);

    var big_int_limbs: std.ArrayList(BigIntLimb) = .empty;
    defer big_int_limbs.deinit(gpa);

    var scratch: std.ArrayList(u32) = .empty;
    defer scratch.deinit(gpa);

    var states: std.ArrayList(State) = .empty;
    defer states.deinit(gpa);

    const gen: Generator = .{
        .tokens = tokens,
        .ast = ast,

        .insts = &insts,
        .extra_data = &extra_data,
        .big_int_limbs = &big_int_limbs,

        .scratch = &scratch,
        .states = &states,
    };

    const root_node = ast.nodes.get(Ast.Node.Index.toIntAllowRoot(.root));
    const root_block = root_node.unpack().block;
    const root_statements = root_block.getNodes(ast.extra_data);

    try gen.insts.ensureUnusedCapacity(gpa, root_statements.len);
    try gen.states.ensureUnusedCapacity(gpa, root_statements.len);
    for (root_statements) |root_statement| {
        gen.states.appendAssumeCapacity(.{ .handle_statement_or_expr = .{
            .node = root_statement,
            .dst = gen.addInstAssumeCapacityUndef(),
        } });
    }
    const root_end: u32 = @intCast(gen.insts.len);

    try gen.runStates(gpa);

    var insts_final = insts.toOwnedSlice();
    errdefer insts_final.deinit(gpa);

    const extra_data_final = try extra_data.toOwnedSlice(gpa);
    errdefer gpa.free(extra_data_final);

    const big_int_limbs_final = try big_int_limbs.toOwnedSlice(gpa);
    errdefer gpa.free(big_int_limbs_final);

    return .{
        .insts = insts_final,
        .root_end = root_end,
        .extra_data = extra_data_final,
        .big_int_limbs = big_int_limbs_final,
    };
}

const State = union(enum) {
    handle_statement_or_expr: struct {
        dst: Inst.Index,
        node: Ast.Node.Index,

        fn init(dst: Inst.Index, node: Ast.Node.Index) @This() {
            return .{ .dst = dst, .node = node };
        }
    },
    expect_expr: struct {
        dst: Inst.Index,
        node: Ast.Node.Index,

        fn init(dst: Inst.Index, node: Ast.Node.Index) @This() {
            return .{ .dst = dst, .node = node };
        }
    },
    handle_block: struct {
        dst: Inst.Index,
        node: Ast.Node.Index,
        index: u32,
        scratch_start: u32,

        fn forNext(state: @This()) @This() {
            var copy = state;
            copy.index += 1;
            return copy;
        }
    },
};

const Generator = struct {
    tokens: Tokens,
    ast: Ast,

    insts: *Inst.Packed.List,
    extra_data: *std.ArrayList(u32),
    big_int_limbs: *std.ArrayList(BigIntLimb),

    scratch: *std.ArrayList(u32),
    states: *std.ArrayList(State),

    fn addInst(
        gen: Generator,
        gpa: std.mem.Allocator,
        data: Inst,
    ) std.mem.Allocator.Error!Inst.Index {
        try gen.insts.ensureUnusedCapacity(gpa, 1);
        return gen.addInstAssumeCapacity(data);
    }

    fn addInstAssumeCapacity(
        gen: Generator,
        data: Inst,
    ) Inst.Index {
        const index = gen.addInstAssumeCapacityUndef();
        gen.insts.set(index.toInt().?, data);
        return index;
    }

    fn addInstAssumeCapacityUndef(self: Generator) Inst.Index {
        return .fromInt(@intCast(self.insts.addOneAssumeCapacity()));
    }

    fn runStates(
        gen: Generator,
        gpa: std.mem.Allocator,
    ) std.mem.Allocator.Error!void {
        while (gen.states.pop()) |state| switch (state) {
            .handle_statement_or_expr => |soe| {
                try gen.states.append(gpa, .{ .expect_expr = .init(soe.dst, soe.node) });
            },
            .expect_expr => |expr| switch (gen.ast.nodes.get(expr.node.toInt().?).unpack()) {
                .grouped => |grouped| gen.states.appendAssumeCapacity(.{ .handle_statement_or_expr = .{
                    .dst = expr.dst,
                    .node = grouped.expr,
                } }),
                .value_ref => |value_ref| try gen.handleExprValueRef(gpa, expr.dst, value_ref),
                .un_op => |un_op| {
                    const op_tok = gen.tokens.getNonNull(un_op.op);
                    const op_kind = op_tok.kind.toOperator().?;

                    try gen.states.ensureUnusedCapacity(gpa, 1);
                    try gen.insts.ensureUnusedCapacity(gpa, 1);
                    const new_inst = gen.addInstAssumeCapacityUndef();
                    gen.states.appendAssumeCapacity(.{ .handle_statement_or_expr = .init(new_inst, un_op.operand) });
                    const inst_value: Inst = switch (op_kind) {
                        else => std.debug.panic("TODO", .{}),
                        .sub => .{ .negate = .{ .operand = new_inst } },
                        .sub_wrap => .{ .negate_wrap = .{ .operand = new_inst } },
                    };
                    gen.insts.set(expr.dst.toInt().?, .pack(inst_value));
                },
                .bin_op => |bin_op| {
                    const op_tok = gen.tokens.getNonNull(bin_op.op);
                    const op_kind = op_tok.kind.toOperator() orelse std.debug.panic("TODO", .{});
                    try gen.insts.ensureUnusedCapacity(gpa, 2);
                    const lhs_inst = gen.addInstAssumeCapacityUndef();
                    const rhs_inst = gen.addInstAssumeCapacityUndef();
                    try gen.states.appendSlice(gpa, &.{
                        .{ .handle_statement_or_expr = .init(rhs_inst, bin_op.rhs) },
                        .{ .handle_statement_or_expr = .init(lhs_inst, bin_op.lhs) },
                    });

                    switch (op_kind) {
                        .colon => {
                            gen.insts.set(expr.dst.toInt().?, .pack(.{ .typed = .{
                                .operand = lhs_inst,
                                .type = rhs_inst,
                            } }));
                        },

                        .ampersand => std.debug.panic("TODO", .{}),
                        .pipe => std.debug.panic("TODO", .{}),
                        .modulo => std.debug.panic("TODO", .{}),

                        inline //
                        .div,

                        .eq,

                        .lt,
                        .lt_eq,

                        .gt,
                        .gt_eq,

                        .add,
                        .add_wrap,
                        .add_saturate,

                        .sub,
                        .sub_wrap,
                        .sub_saturate,

                        .mul,
                        .mul_wrap,
                        .mul_saturate,
                        => |ikind| {
                            const op_tag = @field(Inst.Tag, @tagName(ikind));
                            gen.insts.set(expr.dst.toInt().?, .pack(@unionInit(Inst, @tagName(op_tag), .{
                                .lhs = lhs_inst,
                                .rhs = rhs_inst,
                            })));
                        },
                    }
                },
                .if_else => |if_else| {
                    const if_true, const if_false = if_else.getBranchNodes(gen.ast.extra_data);

                    try gen.insts.ensureUnusedCapacity(gpa, 3);
                    const cond_inst = gen.addInstAssumeCapacityUndef();
                    const true_branch_inst = gen.addInstAssumeCapacityUndef();

                    switch (if_false) {
                        .null => {
                            gen.insts.set(expr.dst.toInt().?, .pack(.{
                                .if_true = .{
                                    .cond = cond_inst,
                                    .true_branch = true_branch_inst,
                                },
                            }));
                            try gen.states.appendSlice(gpa, &.{
                                .{
                                    .expect_expr = .{
                                        .dst = true_branch_inst,
                                        .node = if_true,
                                    },
                                },
                                .{
                                    .expect_expr = .{
                                        .dst = cond_inst,
                                        .node = if_else.cond,
                                    },
                                },
                            });
                        },
                        _ => {
                            const false_branch_inst = gen.addInstAssumeCapacityUndef();
                            const extra_data_start: u32 = @intCast(gen.extra_data.items.len);
                            try gen.extra_data.appendSlice(gpa, &.{
                                true_branch_inst.toInt().?,
                                false_branch_inst.toInt().?,
                            });
                            gen.insts.set(expr.dst.toInt().?, .pack(.{
                                .if_else = .{
                                    .cond = cond_inst,
                                    .branches = extra_data_start,
                                },
                            }));

                            try gen.states.appendSlice(gpa, &.{
                                .{
                                    .expect_expr = .{
                                        .dst = false_branch_inst,
                                        .node = if_false,
                                    },
                                },
                                .{
                                    .expect_expr = .{
                                        .dst = true_branch_inst,
                                        .node = if_true,
                                    },
                                },
                                .{
                                    .expect_expr = .{
                                        .dst = cond_inst,
                                        .node = if_else.cond,
                                    },
                                },
                            });
                        },
                    }
                },
                .block => {
                    try gen.states.append(gpa, .{ .handle_block = .{
                        .dst = expr.dst,
                        .node = expr.node,
                        .index = 0,
                        .scratch_start = @intCast(gen.scratch.items.len),
                    } });
                },
            },
            .handle_block => |block_state| blk: {
                const block = gen.ast.nodes.get(block_state.node.toInt().?).unpack().block;
                const block_nodes = block.getNodes(gen.ast.extra_data);
                if (block_state.index == block_nodes.len) {
                    const extra_start: u32 = @intCast(gen.extra_data.items.len);
                    try gen.extra_data.appendSlice(gpa, gen.scratch.items[block_state.scratch_start..]);
                    const extra_end: u32 = @intCast(gen.extra_data.items.len);
                    gen.scratch.shrinkRetainingCapacity(block_state.scratch_start);
                    gen.insts.set(block_state.dst.toInt().?, .pack(.{ .block = .{
                        .start = extra_start,
                        .end = extra_end,
                    } }));
                    break :blk;
                }
                try gen.states.ensureUnusedCapacity(gpa, 2);
                try gen.scratch.ensureUnusedCapacity(gpa, 1);
                try gen.insts.ensureUnusedCapacity(gpa, 1);
                const soe_inst = gen.addInstAssumeCapacityUndef();
                gen.scratch.appendAssumeCapacity(soe_inst.toInt().?);
                gen.states.appendSliceAssumeCapacity(&.{
                    .{ .handle_block = block_state.forNext() },
                    .{ .handle_statement_or_expr = .init(soe_inst, block_nodes[block_state.index]) },
                });
            },
        };
    }

    fn handleExprValueRef(
        gen: Generator,
        gpa: std.mem.Allocator,
        dst_inst: Inst.Index,
        value_ref: Ast.Node.ValueRef,
    ) std.mem.Allocator.Error!void {
        switch (value_ref.getKind(gen.tokens)) {
            .ident => std.debug.panic("TODO", .{}),
            .number => {
                const full_src = value_ref.getSrc(gen.tokens);
                const is_neg = full_src[0] == '-';
                const type_suffix_start_opt = std.mem.lastIndexOfAny(u8, full_src, &.{ 'u', 'i' });
                const main_src_signed = full_src[0 .. type_suffix_start_opt orelse full_src.len];
                const main_src = full_src[if (is_neg) 1 else 0 .. type_suffix_start_opt orelse full_src.len];

                const int_inst: Inst.Index = if (type_suffix_start_opt) |type_suffix_start| blk: {
                    const type_suffix = full_src[type_suffix_start..];
                    try gen.insts.ensureUnusedCapacity(gpa, 1);
                    const new_inst = gen.addInstAssumeCapacityUndef();
                    gen.insts.set(dst_inst.toInt().?, .pack(.{ .typed = .{
                        .operand = new_inst,
                        .type = if (true) std.debug.panic("TODO: inst for types using {s}", .{type_suffix}),
                    } }));
                    break :blk new_inst;
                } else dst_inst;

                switch (std.zig.parseNumberLiteral(main_src)) {
                    .int => |raw| {
                        if (!is_neg) {
                            gen.insts.set(int_inst.toInt().?, .pack(.{ .int_pos = raw }));
                            return;
                        }
                        if (std.math.negateCast(raw)) |value| {
                            gen.insts.set(int_inst.toInt().?, .pack(.{ .int_neg = value }));
                            return;
                        } else |err| switch (err) {
                            error.Overflow => {}, // fallthrough to int_big branch
                        }
                        std.debug.panic("TODO", .{});
                    },
                    .big_int => |base| {
                        const base_value = @intFromEnum(base);

                        const limb_count = std.math.cast(
                            u31,
                            big_int_math.calcSetStringLimbCount(base_value, main_src.len),
                        ) orelse std.debug.panic("TODO", .{});
                        const limbs_tmp_buf_len = big_int_math.calcSetStringLimbsBufferLen(base_value, main_src.len);

                        try gen.big_int_limbs.ensureUnusedCapacity(gpa, limb_count + limbs_tmp_buf_len);

                        const big_int_limbs_start: u32 = @intCast(gen.big_int_limbs.items.len);
                        const limbs = gen.big_int_limbs.addManyAsSliceAssumeCapacity(limb_count);

                        var big_int: big_int_math.Mutable = .init(limbs, 0);

                        {
                            const limbs_tmp_buffer = gen.big_int_limbs.addManyAsSliceAssumeCapacity(limbs_tmp_buf_len);
                            big_int.setString(base_value, main_src, limbs_tmp_buffer, gpa) catch |err| switch (err) {
                                error.InvalidCharacter => std.debug.panic("TODO", .{}),
                            };
                            gen.big_int_limbs.shrinkRetainingCapacity(gen.big_int_limbs.items.len - limbs_tmp_buf_len);
                        }

                        gen.insts.set(int_inst.toInt().?, .pack(.{ .int_big = .{
                            .start = big_int_limbs_start,
                            .len = limb_count,
                            .positive = !is_neg,
                        } }));
                    },
                    .float => |float_base| {
                        _ = float_base;
                        const float_value = std.fmt.parseFloat(f128, main_src_signed) catch |err| switch (err) {
                            error.InvalidCharacter => std.debug.panic("TODO", .{}),
                        };
                        const start = gen.extra_data.items.len;
                        const buffer = try gen.extra_data.addManyAsArray(gpa, @divExact(@sizeOf(u128), @sizeOf(u32)));
                        std.mem.writeInt(u128, @ptrCast(buffer), @bitCast(float_value), .little);
                        gen.insts.set(int_inst.toInt().?, .pack(.{ .float = .{
                            .start = start,
                        } }));
                    },
                    .failure => std.debug.panic("TODO", .{}),
                }
            },
        }
    }
};

test Ir {
    const gpa = std.testing.allocator;

    const tokens: Tokens = try .tokenizeSlice(gpa,
        \\{
        \\    (32 + 1) * -3;
        \\}
    );
    defer tokens.deinit(gpa);

    const ast: Ast = try .parse(gpa, tokens);
    defer ast.deinit(gpa);

    const ir: Ir = try .generate(gpa, tokens, ast);
    defer ir.deinit(gpa);
}
