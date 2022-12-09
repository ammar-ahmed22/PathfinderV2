import React from "react";
import { TourProvider } from "@reactour/tour";
import {
  useToken,
  useColorModeValue,
} from "@chakra-ui/react";
import { steps } from "./steps";

interface TourProps {
  children: React.ReactNode;
}
const Tour: React.FC<TourProps> = ({ children }) => {
  const [brandPurple500, gray800, white] = useToken("colors", [
    "brand.purple.500",
    "gray.800",
    "white",
  ]);
  const [xlBorderRadius] = useToken("radii", ["xl"]);
  const [spacing8] = useToken("spacing", ["space.8"]);

  const bg = useColorModeValue(white, gray800);
  const text = useColorModeValue(gray800, white);

  return (
    <TourProvider
      steps={steps}
      styles={{
        badge: (base) => ({
          ...base,
          background: brandPurple500,
        }),
        popover: (base) => ({
          ...base,
          background: bg,
          color: text,
          borderRadius: xlBorderRadius,
          padding: spacing8,
          marginLeft: "1rem",
          marginTop: "1rem",
        }),
        dot: (base, state) => {
          console.log(base, state);
          return {
            ...base,
            background: state?.current ? brandPurple500 : base.background,
          };
        },
      }}
      showDots={false}
    >
      {children}
    </TourProvider>
  );
};

export default Tour;
