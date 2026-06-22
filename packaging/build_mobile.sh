#!/usr/bin/env bash
# Capacitor mobile thin-client (Android APK / iOS archive). The app is a PWA pointing at
# a remote Crucible node. All @capacitor/* are pinned to ONE version to avoid Gradle
# duplicate-class conflicts. Best-effort: the PWA is the primary mobile path.
set -euo pipefail
cd "$(dirname "$0")/../frontend"
CAP=7.4.3
npm ci
npm run build
npm install --save-exact \
  "@capacitor/core@${CAP}" "@capacitor/cli@${CAP}" "@capacitor/android@${CAP}" "@capacitor/ios@${CAP}"
[ -f capacitor.config.ts ] || npx --yes cap init Crucible ai.crucible.app --web-dir dist

if [ "${1:-android}" = "android" ]; then
  rm -rf android
  npx --yes cap add android
  npx --yes cap sync android
  cd android
  chmod +x ./gradlew
  ./gradlew --no-daemon assembleDebug
else
  rm -rf ios
  npx --yes cap add ios
  npx --yes cap sync ios
  cd ios/App
  xcodebuild -workspace App.xcworkspace -scheme App \
    -archivePath build/App.xcarchive archive CODE_SIGNING_ALLOWED=NO
fi
