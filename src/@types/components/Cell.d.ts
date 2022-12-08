import Node from "../../helpers/Node";
import { AlgorithmParams } from "../helpers/Node";
export type CornerType = "tl" | "tr" | "br" | "bl";

export interface CellProps {
  node: Node<AlgorithmParams>;
  corner: CornerType | undefined;
}
