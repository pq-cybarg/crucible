import { describe, expect, it } from "vitest";
import { moodFromText } from "./sentiment";

describe("moodFromText — context-free chat sentiment → companion mood", () => {
  it("strong affection → heart eyes above the intensity gate", () => {
    const h = moodFromText("I love this! ❤️");
    expect(h?.mood).toBe("lovestruck");
    expect(h!.weight).toBeGreaterThan(0.55);
  });
  it("hype → star eyes", () => {
    expect(moodFromText("That's incredible!")?.mood).toBe("starstruck");
    expect(moodFromText("wow!!! 🤩")?.mood).toBe("starstruck");
  });
  it("mirth → laughing", () => expect(moodFromText("haha that's funny")?.mood).toBe("laughing"));
  it("confusion → dizzy", () => expect(moodFromText("I'm confused about this")?.mood).toBe("dizzy"));
  it("apology → sad (soft, below the shape gate)", () => {
    const h = moodFromText("Sorry, that failed.");
    expect(h?.mood).toBe("sad");
    expect(h!.weight).toBeLessThan(0.6);
  });
  it("question → curious", () => expect(moodFromText("Which one do you want?")?.mood).toBe("curious"));
  it("flat/neutral text → no forced mood", () => expect(moodFromText("The file has 12 lines.")).toBeNull());

  it("admiration → sparkle", () => expect(moodFromText("That's absolutely gorgeous!")?.mood).toBe("sparkly"));
  it("shock → dot eyes", () => expect(moodFromText("Wait what?! oh no")?.mood).toBe("shock"));
  it("grief → crying", () => expect(moodFromText("that's so sad 😭")?.mood).toBe("crying"));
  it("dead-tired → ko", () => expect(moodFromText("I'm so done, exhausted 💀")?.mood).toBe("ko"));
  it("cat vibes → cat mode", () => expect(moodFromText("meow~ 🐱")?.mood).toBe("cat"));
});
