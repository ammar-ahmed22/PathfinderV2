import type { Keyframes } from "@emotion/react";
import type { CornerType } from "../@types/components/Cell";

export const borderRadii = (corner: CornerType | undefined): object => {
    return {
        borderTopLeftRadius: corner === "tl" ? "xl" : "none",
        borderTopRightRadius: corner === "tr" ? "xl" : "none",
        borderBottomLeftRadius: corner === "bl" ? "xl" : "none",
        borderBottomRightRadius: corner === "br" ? "xl" : "none",
    };
};



export const createAnimation = (
    keyframes: Keyframes,
    duration: number,
    easing: string
) => {
    return `${keyframes} ${duration}s ${easing}`;
};
