const std = @import("std");
const mre = @import("mre");

const Cmd = struct {
    input: Input,

    const Input = union(enum) {
        const Tag = @typeInfo(Input).@"union".tag_type.?;
        /// Indicates desire to run the REPL.
        repl,
        /// Expression to evaluate directly.
        eval: []const u8,
        /// Path to file with source to evaluate.
        run: []const u8,

        pub fn format(
            input: Input,
            w: *std.Io.Writer,
        ) std.Io.Writer.Error!void {
            var sz: std.zon.Serializer = .{ .writer = w };
            switch (input) {
                .repl => try sz.ident("repl"),
                inline .eval, .run => |str, tag| {
                    var struct_sz = try sz.beginStruct(.{
                        .whitespace_style = .{ .wrap = false },
                    });
                    try struct_sz.field(@tagName(tag), str, .{});
                    try struct_sz.end();
                },
            }
        }
    };

    pub fn format(
        cmd: Cmd,
        w: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        var sz: std.zon.Serializer = .{ .writer = w };
        var struct_sz = try sz.beginStruct(.{
            .whitespace_style = .{ .wrap = true },
        });
        try struct_sz.fieldPrefix("input");
        try cmd.input.format(w);
        try struct_sz.end();
    }

    const ParseError = error{
        MissingInput,
        UnexpectedPositional,
        UnexpectedDash,
    };

    fn parse(args: []const []const u8) ParseError!Cmd {
        var args_iter: ArgIter = .{
            .args = args,
            .index = 0,
        };

        const whitespace = std.ascii.whitespace;

        var input: ?Input = null;

        while (args_iter.next()) |tok_untrimmed| {
            const tok = std.mem.trim(u8, tok_untrimmed, &whitespace);
            if (tok.len == 0) continue;
            if (std.mem.startsWith(u8, tok, "-")) {
                return error.UnexpectedDash;
            } else {
                if (input != null) return error.UnexpectedPositional;
                const name = std.meta.stringToEnum(Input.Tag, tok) orelse return error.UnexpectedPositional;
                input = switch (name) {
                    .repl => .repl,
                    inline .eval, .run => |itag| @unionInit(
                        Input,
                        @tagName(itag),
                        args_iter.next() orelse return error.MissingInput,
                    ),
                };
            }
        }

        return .{
            .input = input orelse return error.MissingInput,
        };
    }

    const ArgIter = struct {
        args: []const []const u8,
        index: usize,

        fn peek(self: *ArgIter) ?[]const u8 {
            if (self.index == self.args.len) return null;
            return self.args[self.index];
        }

        fn next(self: *ArgIter) ?[]const u8 {
            if (self.index == self.args.len) return null;
            defer self.index += 1;
            return self.args[self.index];
        }
    };
};

pub fn main() !void {
    var dba_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = dba_state.deinit();
    const dba = dba_state.allocator();
    const gpa = dba;

    const argv = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, argv);

    const cmd: Cmd = try .parse(argv[1..]);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_fw: std.fs.File.Writer = .init(.stdout(), &stdout_buffer);
    const stdout = &stdout_fw.interface;
    _ = stdout;

    var stderr_buffer: [4096]u8 = undefined;
    var stderr_fw: std.fs.File.Writer = .init(.stderr(), &stderr_buffer);
    const stderr = &stderr_fw.interface;

    try stderr.print("cmd: {f}\n", .{cmd});
    try stderr.flush();

    switch (cmd.input) {
        inline .eval, .run => |expr_or_path, input_tag| {
            const tokens: mre.Tokens = switch (input_tag) {
                .repl => comptime unreachable,
                .eval => try .tokenizeSlice(gpa, expr_or_path),
                .run => run: {
                    const src_file = try std.fs.cwd().openFile(expr_or_path, .{});
                    defer src_file.close();

                    var buffer: [4096]u8 = undefined;
                    var src_file_reader = src_file.reader(&buffer);
                    break :run try .tokenizeReader(gpa, &src_file_reader.interface);
                },
            };
            defer tokens.deinit(gpa);
            try printTokensDump(stderr, tokens);
            try stderr.flush();

            const ast = try mre.Ast.parse(gpa, tokens);
            defer ast.deinit(gpa);
            try printAstDump(stderr, tokens, ast);
            try stderr.flush();

            const ir: mre.Ir = try .generate(gpa, tokens, ast);
            defer ir.deinit(gpa);
            try printIrDump(gpa, stderr, ir);
            try stderr.flush();
        },
        .repl => {
            std.log.err("TODO: implement repl", .{});
            return;
        },
    }
}

fn printTokensDump(
    w: *std.Io.Writer,
    tokens: mre.Tokens,
) std.Io.Writer.Error!void {
    const kind_longest_index = findLongestEnumTagOf(mre.Lexer.Token.Kind, tokens.list.items(.kind));
    const kind_width = @tagName(tokens.list.items(.kind)[kind_longest_index]).len;

    const start_max: usize, //
    const end_max: usize //
    = max: {
        var start_max: usize = 0;
        var end_max: usize = 0;
        for (tokens.tokenLocs()) |loc| {
            start_max = @max(start_max, loc.start);
            end_max = @max(end_max, loc.end);
        }
        break :max .{ start_max, end_max };
    };

    const tokens_start_width = std.fmt.count("{d}", .{start_max});
    const tokens_end_width = std.fmt.count("{d}", .{end_max});

    const index_width = std.fmt.count("{d}", .{tokens.list.len -| 1});
    try w.writeAll("tokens: .{");
    for (0..tokens.list.len) |i| {
        const token_index: mre.Tokens.Value.Index = .fromInt(@intCast(i));
        const token = tokens.get(token_index).?;

        if (i == 0) try w.writeAll("\n");
        try w.print(
            "    [{[index]d: >[index_width]}]: .{{ {[kind]t: <[kind_width]}," ++
                (" .{{ " ++
                    "{[loc_start]d: <[loc_start_width]}, " ++
                    "{[loc_end]d: <[loc_end_width]} }}") ++
                ", '{[str]f}' }},\n",
            .{
                .index = i,
                .index_width = index_width,

                .kind = token.kind,
                .kind_width = kind_width,

                .loc_start = token.loc.start,
                .loc_start_width = tokens_start_width,

                .loc_end = token.loc.end,
                .loc_end_width = tokens_end_width,

                .str = std.zig.fmtString(token.loc.getSrc(tokens.src.slice())),
            },
        );
    }
    try w.writeAll("}\n");
}

fn printAstDump(
    w: *std.Io.Writer,
    tokens: mre.Tokens,
    ast: mre.Ast,
) std.Io.Writer.Error!void {
    const tag_longest_index = findLongestEnumTagOf(mre.Ast.Node.Tag, ast.nodes.items(.tag));
    const tag_width = @tagName(ast.nodes.items(.tag)[tag_longest_index]).len;

    try w.writeAll("ast: .{\n");
    try printExtraDataDump(w, 1, ast.extra_data);

    try w.writeAll("    .nodes = .{");
    if (ast.nodes.len != 0) {
        const index_width = std.fmt.count("{d}", .{ast.nodes.len -| 1});

        for (0..ast.nodes.len) |i| {
            const node_index: mre.Ast.Node.Index = .fromInt(@intCast(i));
            const node_packed = ast.get(node_index);
            const node = node_packed.unpack();
            try w.writeAll("\n");
            try w.splatByteAll(' ', 4 * 2);
            try w.print("[{[index]d: >[index_width]}]: .{{ .{[tag]t: <[tag_width]} = ", .{
                .tag = node_packed.tag,
                .tag_width = tag_width,

                .index = i,
                .index_width = index_width,
            });
            switch (node) {
                .value_ref => |value_ref| try w.print(
                    ".{{ .main_token = {[index]d} }}",
                    .{ .index = value_ref.main_token.toInt().? },
                ),
                .un_op => |un_op| try w.print(
                    ".{{ .op = {[op]d}, .value = {[operand]d} }}",
                    .{
                        .op = un_op.op.toInt().?,
                        .operand = un_op.operand.toInt().?,
                    },
                ),
                .bin_op => |bin_op| try w.print(
                    ".{{ .lhs = {[lhs]d}, .op = {[op]d}, .rhs = {[rhs]d} }}",
                    .{
                        .lhs = bin_op.lhs.toInt().?,
                        .op = bin_op.op.toInt().?,
                        .rhs = bin_op.rhs.toInt().?,
                    },
                ),
                .if_else => |if_else| {
                    const true_branch, //
                    const false_branch //
                    = if_else.getBranchNodes(ast.extra_data);
                    try w.print(
                        ".{{ .cond = {[cond]d}, .true = {[true]d}, .false = {[false]?d} }}",
                        .{
                            .cond = if_else.cond.toInt().?,
                            .true = true_branch.toInt().?,
                            .false = false_branch.toInt(),
                        },
                    );
                },
                .grouped => |grouped| try w.print(
                    ".{{ .paren_l = {[paren_l]d}, .paren_r = {[paren_r]d}, .expr = {[expr]d} }}",
                    .{
                        .paren_l = grouped.paren_l.toInt().?,
                        .paren_r = grouped.paren_r.toInt().?,
                        .expr = grouped.expr.toInt().?,
                    },
                ),
                .block => |block| try w.print(
                    ".{{ .start = {}, .end = {} }}",
                    .{
                        block.start,
                        block.end,
                    },
                ),
            }

            try w.writeAll(" },");
        }
        try w.writeByte('\n');
        try w.splatByteAll(' ', 4 * 1);
    }
    try w.writeAll("},\n");
    try w.writeAll("}\n");

    var walk_buffer: [32]u64 = undefined;
    try w.print("nodeFmt(ast) = {f}\n", .{ast.nodeFmt(tokens, .{
        .node = .root,
        .walk_buffer = &walk_buffer,
        .options = .default_prec_delim,
    })});
}

fn printIrDump(
    gpa: std.mem.Allocator,
    w: *std.Io.Writer,
    ir: mre.Ir,
) std.Io.Writer.Error!void {
    const tag_longest_index = findLongestEnumTagOf(mre.Ir.Inst.Tag, ir.insts.items(.tag));
    const tag_width = @tagName(ir.insts.items(.tag)[tag_longest_index]).len;

    try w.writeAll("ir: {\n");
    try printExtraDataDump(w, 1, ir.extra_data);

    try w.writeAll("    .insts = .{");
    if (ir.insts.len != 0) {
        const index_width = std.fmt.count("{d}", .{ir.insts.len -| 1});

        var big_int_str_buf: std.ArrayList(u8) = .empty;
        defer big_int_str_buf.deinit(gpa);

        var big_int_str_limbs_buf: std.ArrayList(std.math.big.Limb) = .empty;
        defer big_int_str_limbs_buf.deinit(gpa);

        for (0..ir.insts.len) |i| {
            const inst_index: mre.Ir.Inst.Index = .fromInt(@intCast(i));
            const inst_packed = ir.insts.get(inst_index.toInt().?);
            const inst = inst_packed.unpack();
            try w.writeAll("\n");
            try w.splatByteAll(' ', 4 * 2);
            try w.print("[{[index]d: >[index_width]}]: .{{ .{[tag]t: <[tag_width]} = ", .{
                .tag = inst,
                .tag_width = tag_width,

                .index = i,
                .index_width = index_width,
            });

            switch (inst) {
                inline .int_pos, .int_neg => |int| try w.print(".{{ .value = {d} }}", .{int}),
                .int_big => |int_big| {
                    const big_int = int_big.getBigIntConst(ir.big_int_limbs);
                    try w.writeAll(".{ . value = ");
                    try big_int.formatNumber(w, .{});
                    try w.writeAll("}");
                },
                .float => |float| try w.print("{d}", .{float.get(ir.extra_data)}),

                .typed => |typed| try w.print(".{{ {d}, {d} }}", .{
                    typed.operand.toInt().?,
                    typed.type.toInt().?,
                }),

                .negate, .negate_wrap => |negate| try w.print(
                    ".{{ .operand = {d} }}",
                    .{negate.operand.toInt().?},
                ),

                .eq,
                .lt,
                .lt_eq,
                .gt,
                .gt_eq,

                .div,

                .add,
                .add_wrap,
                .add_saturate,

                .sub,
                .sub_wrap,
                .sub_saturate,

                .mul,
                .mul_wrap,
                .mul_saturate,
                => |bin_op| try w.print(
                    ".{{ .lhs = {d}, .rhs = {d} }}",
                    .{ bin_op.lhs.toInt().?, bin_op.rhs.toInt().? },
                ),

                .if_true => |if_true| try w.print(
                    ".{{ .cond = {d}, .if_true = {d} }}",
                    .{ if_true.cond.toInt().?, if_true.true_branch.toInt().? },
                ),
                .if_else => |if_else| {
                    const if_true, const if_false = if_else.getBranches(ir.extra_data).*;
                    try w.print(
                        ".{{ .cond = {d}, .if_true = {d}, .if_false = {d} }}",
                        .{ if_else.cond.toInt().?, if_true.toInt().?, if_false.toInt().? },
                    );
                },

                .block => |block| try w.print(
                    ".{{ .start = {d}, .end = {d} }}",
                    .{ block.start, block.end },
                ),
            }

            try w.writeAll(" },");
        }
        try w.writeByte('\n');
        try w.splatByteAll(' ', 4 * 1);
    }
    try w.writeAll("},\n");
    try w.writeAll("}\n");
}

fn printExtraDataDump(
    w: *std.Io.Writer,
    indent: usize,
    extra_data: []const u32,
) std.Io.Writer.Error!void {
    try w.splatByteAll(' ', 4 * (indent + 0));
    try w.writeAll(".extra_data = .{");
    if (extra_data.len != 0) {
        try w.writeByte('\n');
        const window_size: u32 = blk: {
            const widths = [_]u32{ 32, 28, 26, 24, 20, 18, 16, 15, 12, 10, 9, 8, 6, 5, 4, 3 };
            var closest: u32 = 0;
            for (widths) |width| {
                if (extra_data.len % width == 0) break :blk width;
                if (closest == 0 or (extra_data.len % closest) < extra_data.len % width) {
                    closest = width;
                }
            }
            break :blk closest;
        };
        var window = std.mem.window(u32, extra_data, window_size, window_size);
        while (window.next()) |segment| {
            try w.splatByteAll(' ', 4 * (indent + 1));
            for (segment, 0..) |value, i| {
                if (i != 0) try w.writeByte(' ');
                try w.print("{d},", .{value});
            }
            try w.writeByte('\n');
        }
        try w.splatByteAll(' ', 4 * (indent + 0));
    }
    try w.writeAll("},\n");
}

fn findLongestEnumTagOf(comptime E: type, tags: []const E) usize {
    var tag_longest_index: usize = 0;
    for (tags, 0..) |tag, i| {
        const tag_longest = tags[tag_longest_index];
        if (@tagName(tag_longest).len >= @tagName(tag).len) continue;
        tag_longest_index = i;
    }
    return tag_longest_index;
}
