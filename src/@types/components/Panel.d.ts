import * as React from "react";

export interface PanelProps {
    children: React.ReactNode;
    width?: string;
    bg?: string;
    height?: string;
    styles?: Record<string, string | number | object>;
    customRef?: React.MutableRefObject;
}
