#!/usr/bin/env bash
# Capacitor mobile thin-client (Android APK / iOS archive). The PWA points at a remote
# Crucible node. CI-targeted (android: ubuntu+JDK; ios: macos+Xcode).
set -euo pipefail
cd "$(dirname "$0")/../frontend"
npm ci
npm run build
npm install -D @capacitor/cli @capacitor/core @capacitor/android @capacitor/ios
[ -f capacitor.config.ts ] || npx cap init Crucible ai.crucible.app --web-dir dist
if [ "${1:-android}" = "android" ]; then
  [ -d android ] || npx cap add android
  npx cap sync android
  cd android && chmod +x ./gradlew && ./gradlew assembleDebug
else
  [ -d ios ] || npx cap add ios
  npx cap sync ios
  cd ios/App && xcodebuild -workspace App.xcworkspace -scheme App \
    -archivePath build/App.xcarchive archive CODE_SIGNING_ALLOWED=NO || true
fi
