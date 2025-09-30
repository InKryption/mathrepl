const std = @import("std");
const big_int_math = std.math.big.int;

const builtin = @import("builtin");

const mre = @import("mre.zig");
const Lexer = mre.Lexer;
const Tokens = mre.Tokens;
const Ast = mre.Ast;

const BigIntLimb = std.math.big.Limb;

const Ir = @This();
insts: Inst.List.Slice,
root_end: u32,
extra_data: []const u32,
big_int_limbs: []const BigIntLimb,

pub fn deinit(ir: Ir, gpa: std.mem.Allocator) void {
    var insts = ir.insts;
    insts.deinit(gpa);
    gpa.free(ir.extra_data);
    gpa.free(ir.big_int_limbs);
}

pub const Inst = union(enum(u8)) {
    int_pos: u64,
    int_neg: i64,
    int_big: IntBig,
    float: Float128,

    typed: Typed,

    add: BinOp,
    add_wrap: BinOp,
    add_saturate: BinOp,

    sub: BinOp,
    sub_wrap: BinOp,
    sub_saturate: BinOp,

    mul: BinOp,
    mul_wrap: BinOp,
    mul_saturate: BinOp,

    pub const Tag = @typeInfo(Inst).@"union".tag_type.?;

    comptime {
        const ExternInst = @Type(.{ .@"union" = .{
            .layout = .@"extern",
            .tag_type = null,
            .fields = @typeInfo(Inst).@"union".fields,
            .decls = &.{},
        } });
        if (@sizeOf(ExternInst) != 8) @compileError(
            std.fmt.comptimePrint("Wrong size '{d}'", .{@sizeOf(ExternInst)}),
        );
    }

    pub const List = std.MultiArrayList(Inst);
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

    pub const BinOp = extern struct {
        lhs: Inst.Index,
        rhs: Inst.Index,
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
    var insts: Inst.List = .empty;
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
};

const Generator = struct {
    tokens: Tokens,
    ast: Ast,

    insts: *Inst.List,
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
            .handle_statement_or_expr => |soe| switch (gen.ast.nodes.get(soe.node.toInt().?).unpack()) {
                else => std.debug.panic("TODO", .{}),
                .value_ref => |value_ref| try gen.handleExprValueRef(gpa, soe.dst, value_ref),
                .bin_op => |bin_op| {
                    const op_tok = gen.tokens.list.get(bin_op.op);
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
                            gen.insts.set(soe.dst.toInt().?, .{ .typed = .{
                                .operand = lhs_inst,
                                .type = rhs_inst,
                            } });
                        },

                        .ampersand => std.debug.panic("TODO", .{}),
                        .pipe => std.debug.panic("TODO", .{}),
                        .percent => std.debug.panic("TODO", .{}),
                        .slash => std.debug.panic("TODO", .{}),

                        inline //
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
                            gen.insts.set(soe.dst.toInt().?, @unionInit(Inst, @tagName(op_tag), .{
                                .lhs = lhs_inst,
                                .rhs = rhs_inst,
                            }));
                        },
                    }
                },
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
                    gen.insts.set(dst_inst.toInt().?, .{ .typed = .{
                        .operand = new_inst,
                        .type = if (true) std.debug.panic("TODO: inst for types using {s}", .{type_suffix}),
                    } });
                    break :blk new_inst;
                } else dst_inst;

                switch (std.zig.parseNumberLiteral(main_src)) {
                    .int => |raw| {
                        if (!is_neg) {
                            gen.insts.set(int_inst.toInt().?, .{ .int_pos = raw });
                            return;
                        }
                        if (std.math.negateCast(raw)) |value| {
                            gen.insts.set(int_inst.toInt().?, .{ .int_neg = value });
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

                        gen.insts.set(int_inst.toInt().?, .{ .int_big = .{
                            .start = big_int_limbs_start,
                            .len = limb_count,
                            .positive = !is_neg,
                        } });
                    },
                    .float => |float_base| {
                        _ = float_base;
                        const float_value = std.fmt.parseFloat(f128, main_src_signed) catch |err| switch (err) {
                            error.InvalidCharacter => std.debug.panic("TODO", .{}),
                        };
                        const start = gen.extra_data.items.len;
                        const buffer = try gen.extra_data.addManyAsArray(gpa, @divExact(@sizeOf(u128), @sizeOf(u32)));
                        std.mem.writeInt(u128, @ptrCast(buffer), @bitCast(float_value), .little);
                        gen.insts.set(int_inst.toInt().?, .{ .float = .{
                            .start = start,
                        } });
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
        \\    (32u8 + 1) * 3:u8;
        \\}
    );
    defer tokens.deinit(gpa);

    const ast: Ast = try .parse(gpa, tokens);
    defer ast.deinit(gpa);

    const ir: Ir = try .generate(gpa, tokens, ast);
    defer ir.deinit(gpa);
}
