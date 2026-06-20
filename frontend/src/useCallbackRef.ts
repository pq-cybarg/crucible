import { useCallback, useRef } from "react";

// Stable callback whose identity never changes but always calls the latest fn.
export function useCallbackRef<TArgs extends readonly unknown[], TReturn>(
  fn: (...args: TArgs) => TReturn,
): (...args: TArgs) => TReturn {
  const ref = useRef(fn);
  ref.current = fn;
  return useCallback((...args: TArgs): TReturn => ref.current(...args), []);
}
