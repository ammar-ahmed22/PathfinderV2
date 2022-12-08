import MinPriorityQueue from "./MinPriorityQueue";

test("testing min priority queue pop function", () => {
  const q = new MinPriorityQueue();
  q.insert("Maryam", 0);
  q.insert("Ammar", 1);
  q.insert("Zaryab", 2);
  q.insert("Ghosia", 3);

  console.log("before pop:", q.heap);
  const removed = q.pop();
  console.log("after pop:", q.heap);

  const removed2 = q.pop();
  const removed3 = q.pop();

  expect(removed).toBe("Maryam");
  expect(removed2).toBe("Ammar");
  expect(removed3).toBe("Zaryab");
});
