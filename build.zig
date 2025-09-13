const std = @import("std");
const Build = std.Build;

pub fn build(b: *Build) void {
    const is_root = b.pkg_hash.len == 0;
    const maybe_target = if (is_root) b.standardTargetOptions(.{}) else null;
    const maybe_optimize = if (is_root) b.standardOptimizeOption(.{}) else null;

    const mre_mod = b.addModule("mre", .{
        .root_source_file = b.path("src/mre/mre.zig"),
        .target = maybe_target,
        .optimize = maybe_optimize,
    });

    // local dev code
    if (!is_root) return;
    const target = maybe_target.?;
    const optimize = maybe_optimize.?;
    const bin_opts: BinOptions = .fromBuildOptions(b);
    const filters = b.option([]const []const u8, "filter", "Filters for tests.") orelse &.{};

    const main_step = b.step("main", "Main executable step.");
    const test_step = b.step("test", "Run all tests.");
    const unit_test_step = b.step("unit-test", "Run unit tests.");
    const main_unit_test_step = b.step("main-unit-test", "Run main unit tests.");
    const mre_unit_test_step = b.step("mre-unit-test", "Run mre unit tests.");
    const check_step = b.step("check", "Check step."); // mainly for ZLS.

    test_step.dependOn(unit_test_step);
    unit_test_step.dependOn(main_unit_test_step);
    unit_test_step.dependOn(mre_unit_test_step);

    check_step.dependOn(main_step);
    check_step.dependOn(test_step);

    const main_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "mre", .module = mre_mod },
        },
    });

    const main_exe = b.addExecutable(.{
        .name = "mathrepl",
        .root_module = main_mod,
    });
    const main_out = addExeOutputs(b, .{
        .exe = main_exe,
        .step = main_step,
        .bin = bin_opts,
        .install = .{},
    });
    if (main_out.run) |main_run| {
        main_run.addArgs(b.args orelse &.{});
    }

    const main_unit_test_exe = b.addTest(.{
        .name = "main-unit-test",
        .root_module = main_mod,
        .filters = filters,
    });
    const main_unit_test_out = addExeOutputs(b, .{
        .exe = main_unit_test_exe,
        .step = main_unit_test_step,
        .bin = bin_opts,
        .install = .{},
    });
    _ = main_unit_test_out;

    const mre_unit_test_exe = b.addTest(.{
        .name = "mre-unit-test",
        .root_module = mre_mod,
        .filters = filters,
    });
    const mre_unit_test_out = addExeOutputs(b, .{
        .exe = mre_unit_test_exe,
        .step = mre_unit_test_step,
        .bin = bin_opts,
        .install = .{},
    });
    _ = mre_unit_test_out;
}

const BinOptions = packed struct {
    install: bool,
    run: bool,

    fn fromBuildOptions(b: *Build) BinOptions {
        const no_bin = b.option(
            bool,
            "no-bin",
            "Don't install any of the binaries implied by the specified steps.",
        ) orelse false;
        const no_run = b.option(
            bool,
            "no-run",
            "Don't run any of the executables implied by the specified steps.",
        ) orelse false;
        return .{
            .install = !no_bin,
            .run = !no_run,
        };
    }
};

const ExeOutputs = struct {
    install: ?*Build.Step.InstallArtifact,
    run: ?*Build.Step.Run,
};

fn addExeOutputs(
    b: *Build,
    params: struct {
        exe: *Build.Step.Compile,
        step: *Build.Step,
        bin: BinOptions,
        install: Build.Step.InstallArtifact.Options,
    },
) ExeOutputs {
    const exe = params.exe;
    const step = params.step;
    const artifact_opts = params.bin;
    const install_opts = params.install;

    const maybe_exe_install = if (artifact_opts.install) b.addInstallArtifact(exe, install_opts) else null;
    const maybe_exe_run = if (artifact_opts.run) b.addRunArtifact(exe) else null;

    step.dependOn(&exe.step);
    if (maybe_exe_install) |exe_install| step.dependOn(&exe_install.step);
    if (maybe_exe_run) |exe_run| step.dependOn(&exe_run.step);

    const install_step = b.getInstallStep();
    install_step.dependOn(&exe.step);
    if (maybe_exe_install) |exe_install| install_step.dependOn(&exe_install.step);

    return .{
        .install = maybe_exe_install,
        .run = maybe_exe_run,
    };
}
