import * as React from "react";
import { VStack } from "@chakra-ui/react";
import { SideBarProps } from "../@types/components/SideBar";

const SideBar: React.FC<SideBarProps> = ({ children, width }) => {
  return (
    <VStack
      h="calc(100vh - var(--pf-space-5) * 2)"
      w={width ? width : "20vw"}
      spacing={5}
    >
      {children}
    </VStack>
  );
};

export default SideBar;
