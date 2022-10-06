

export default class Vec2{
  constructor(
    public x: number = 0,
    public y: number = 0
  ) {}

  public equals = (v: Vec2) : boolean => (this.x === v.x) && (this.y === v.y);

  static Distance = (a: Vec2, b: Vec2) : number => Math.sqrt(Math.pow(b.x - a.x, 2) + Math.pow(b.y - a.y, 2));

  static DistanceSquared = (a: Vec2, b: Vec2) : number => Math.pow(b.x - a.x, 2) + Math.pow(b.y - a.y, 2);

}