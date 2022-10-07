import React, { useEffect, useContext } from "react";
import { Menu, MenuButton, MenuList, MenuItem, Button, MenuOptionGroup, MenuItemOption } from "@chakra-ui/react";
import { ChevronDownIcon } from "@chakra-ui/icons";
import { StoreContext } from "../Store";

import { isAlgorithm } from "../utils/types";
import { StoreContextType } from "../@types/Store";

const AlgorithmMenu : React.FC = () => {


  const store = useContext(StoreContext) as StoreContextType;

  useEffect(() => {
    store.setSelectedAlgorithm("astar");
  }, [])

  const algorithmMenuNames : Record<string, string> = {
    astar: "A*",
    djikstra: "Djikstra's"
  }

  const handleChange = (val: string | string[]) => {
    if (typeof(val) === "string" && isAlgorithm(val)){
      store.setSelectedAlgorithm(val);
    }
  }

  return (
    <Menu >
      <MenuButton as={Button} rightIcon={<ChevronDownIcon />} >
        Select Algorithm
      </MenuButton>
      <MenuList>
        <MenuOptionGroup defaultValue="astar" title="Algorithms" type="radio" onChange={handleChange} >
          {
            Object.keys(algorithmMenuNames).map( algoName => {
              return (
                <MenuItemOption value={algoName} >
                  {
                    algorithmMenuNames[algoName]
                  }
                </MenuItemOption>
              )
            })
          }
        </MenuOptionGroup>
      </MenuList>
    </Menu>
  )
}

export default AlgorithmMenu;