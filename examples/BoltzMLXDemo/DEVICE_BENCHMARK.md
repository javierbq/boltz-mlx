# On-device benchmark (iPhone 15 Pro)

Two ways to run on device once the int8 model + a feature bundle are pushed into the app
container under `Documents/model` and `Documents/features` (each folder must contain its
own `*.safetensors` + manifests — `devicectl` flattens folder contents, so push to
per-artifact subdirs):

- **Interactive (default build):** launch the app and tap **"Run on-device test (bundled
  weights)"**. It runs the prediction on the pushed artifacts and shows live Phase / Tokens
  / Output atoms / Elapsed / MLX peak in the UI. The pushed folders are also visible in
  **Files → On My iPhone → BoltzMLXDemo** (UIFileSharingEnabled), so the "Predict structure"
  document-picker flow works too.
- **Headless:** `BenchmarkRunner.runIfRequested()` (in `BenchmarkRunner.swift`) scans
  `Documents/` for `model.safetensors`/`features.safetensors`, runs one prediction, and
  writes `Documents/result.json` (peak memory, phys_footprint, wall time, counts). It is no
  longer auto-invoked at launch (to avoid colliding with the interactive button); call it
  from a launch hook / launch argument when automating.

## One-time prerequisite (requires your credentials — cannot be automated)

Sign in to your Apple ID in **Xcode → Settings → Accounts** with a team that can sign
`io.github.javierbq.boltzmlx` (or edit `PRODUCT_BUNDLE_IDENTIFIER` in `project.yml` to a
free-provisioning bundle id under your personal team). This is why the automated run is
blocked: `xcodebuild` reported *"No Account for Team YOUR_TEAM_ID"* and there are no
provisioning profiles installed.

## Run (from repo root)

```bash
UDID=YOUR_DEVICE_UDID          # Javier's iPhone 15 Pro (iPhone16,1)
TEAM=<your-team-id>
BID=io.github.javierbq.boltzmlx
cd examples/BoltzMLXDemo && xcodegen generate

# 1. Build + install (signed)
xcodebuild -project BoltzMLXDemo.xcodeproj -scheme BoltzMLXDemo \
  -destination "platform=iOS,id=$UDID" \
  -allowProvisioningUpdates -skipPackagePluginValidation \
  DEVELOPMENT_TEAM="$TEAM" CODE_SIGN_STYLE=Automatic \
  -derivedDataPath build/dd build
xcrun devicectl device install app --device "$UDID" \
  build/dd/Build/Products/Debug-iphoneos/BoltzMLXDemo.app

# 2. Push the int8 model + a feature bundle into the app container
cd ../..
xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BID" --source .artifacts/boltz2-mlx-int8 --destination Documents/bench/model
xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BID" --source .artifacts/features/prot_no_msa --destination Documents/bench/features

# 3. Launch (auto-runs the benchmark) then read the result back
xcrun devicectl device process launch --device "$UDID" "$BID"
sleep 30
xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
  --domain-identifier "$BID" --source Documents/bench/result.json --destination ./device_result.json
cat device_result.json
```

`result.json` fields: `device`, `os_version`, `tokens`, `atoms`, `elapsed_seconds`,
`mlx_peak_bytes` (MLX-tracked peak), `phys_footprint_after_load_bytes`,
`phys_footprint_after_predict_bytes` (jetsam-relevant process footprint).

Mac reference for the same input (`prot_no_msa`, 117 tok / 899 atoms): ~1.66 GB peak
footprint, 504 MB int8 weights. The device run confirms the iPhone 15 Pro figure.
