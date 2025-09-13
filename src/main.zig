const std = @import("std");
const mre = @import("mre");

const Cmd = struct {
    input: Input,

    const Input = union(enum) {
        const Tag = @typeInfo(Input).@"union".tag_type.?;
        repl,
        eval: []const u8,
        run: []const u8,
    };

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
    std.log.info("{}", .{cmd});

    switch (cmd.input) {
        .repl => {
            std.log.err("TODO: implement repl", .{});
            return;
        },
        .eval => {
            
        },
        .run => {},
    }
}
