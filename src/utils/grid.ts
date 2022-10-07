import Vec2 from "../helpers/Vec2";

import type { CornerType } from "../@types/components/Cell";

export const isCorner = (index: Vec2, rows: number, cols: number) : CornerType | undefined => {
  const corners = {
    topLeft: new Vec2(),
    topRight: new Vec2(cols - 1, 0),
    bottomLeft: new Vec2(0, rows - 1),
    bottomRight: new Vec2(cols - 1, rows - 1)
  }

  if (index.equals(corners.topLeft)) return "tl";
  if (index.equals(corners.topRight)) return "tr";
  if (index.equals(corners.bottomLeft)) return "bl";
  if (index.equals(corners.bottomRight)) return "br";
  
  return undefined
}