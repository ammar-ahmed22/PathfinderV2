export const sleep = (ms: number): Promise<boolean> =>
  new Promise((resolve) => setTimeout(resolve, ms));
