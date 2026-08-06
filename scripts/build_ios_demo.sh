#!/usr/bin/env bash
set -euo pipefail

mode="${1:-simulator}"
root="$(cd "$(dirname "$0")/.." && pwd)"
demo="$root/examples/BoltzMLXDemo"

cd "$demo"
xcodegen generate

case "$mode" in
  simulator)
    destination="generic/platform=iOS Simulator"
    sdk="iphonesimulator"
    ;;
  device)
    destination="generic/platform=iOS"
    sdk="iphoneos"
    ;;
  *)
    echo "usage: $0 [simulator|device]" >&2
    exit 2
    ;;
esac

xcodebuild \
  -project BoltzMLXDemo.xcodeproj \
  -scheme BoltzMLXDemo \
  -configuration Release \
  -sdk "$sdk" \
  -destination "$destination" \
  -skipPackagePluginValidation \
  CODE_SIGNING_ALLOWED=NO \
  build
