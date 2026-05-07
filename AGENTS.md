# Repository Notes

- Keep the plugin version in `.claude-plugin/plugin.json` and `.claude-plugin/marketplace.json` identical. When one is bumped, update the other in the same change.
- Release binaries get their version from the Git tag via Go `-ldflags`. When cutting a release, keep the tag and plugin version aligned on the same semver.
