import * as React from "react";
import { IconType } from "react-icons";

export interface PanelProps {
    children: React.ReactNode;
    width?: string;
    bg?: string;
    height?: string;
    styles?: Record<string, string | number | object>;
    customRef?: React.MutableRefObject;
    accordion?: boolean;
    heading?: string;
    headingIcon?: IconType;
    accordionDefaultOpen?: boolean;
}
