import React, { useContext } from "react";
import {
  Menu,
  MenuButton,
  MenuList,
  Button,
  MenuOptionGroup,
  MenuItemOption,
} from "@chakra-ui/react";
import { ChevronDownIcon } from "@chakra-ui/icons";

import { StoreContext } from "../Store";
import type { StoreContextType } from "../@types/Store";

export const speedMenuNames: Record<string, string> = {
  10: "Fast",
  100: "Medium",
  200: "Slow",
};

const SpeedMenu: React.FC = () => {
  const store = useContext(StoreContext) as StoreContextType;

  const handleChange = (val: string | string[]) => {
    if (typeof val === "string") {
      store.setVisualDelay(parseInt(val));
    }
  };

  return (
    <Menu matchWidth>
      <MenuButton
        as={Button}
        rightIcon={<ChevronDownIcon />}
        variant="brandPurple"
        size="sm"
      >
        Speed
      </MenuButton>
      <MenuList>
        <MenuOptionGroup
          defaultValue={"10"}
          title="Speeds"
          type="radio"
          onChange={handleChange}
        >
          {Object.keys(speedMenuNames).map((speed) => {
            return (
              <MenuItemOption value={speed}>
                {speedMenuNames[speed]}
              </MenuItemOption>
            );
          })}
        </MenuOptionGroup>
      </MenuList>
    </Menu>
  );
};

export default SpeedMenu;
