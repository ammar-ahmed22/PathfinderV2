import Vec2 from "../helpers/Vec2";
import { MazeGenerator } from "./maze";

// test("testing maze subdivision", () => {
//   const m = new MazeGenerator(new Vec2(21, 21));
//   m.log()
//   expect(true).toBe(true)
// })

test("testing if array with row, col can check if value is included", () => {
  const testArr: number[][] = [
    [0, 0],
    [1, 1],
    [3, 4],
    [2, 2],
  ];

  console.log(testArr.includes([3, 4]));
  expect(testArr.includes([3, 4])).toBe(true);
  expect(testArr.includes([1, 2])).toBe(false);
});

test("testing generate maze", () => {
  const m = new MazeGenerator(new Vec2(15, 15));
  //m.log();
  m.generate();
  m.log();
  expect(true).toBe(true);
});
