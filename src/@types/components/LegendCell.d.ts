import type { IconType } from "react-icons";

export type LegendCellProps = {
  size: string;
  bg?: string | "borderColor";
  bgGradient?: string;
  borderWidth?: string;
  cellName: string;
  icon?: IconType;
};
