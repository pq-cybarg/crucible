import type { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId: "ai.crucible.app",
  appName: "Crucible",
  webDir: "dist",
  server: { androidScheme: "https" },
};

export default config;
