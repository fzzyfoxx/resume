import React, { useState, useEffect } from 'react';

import useFilterChains from '../../hooks/useFilterChains';
import FilterChainAccordion2 from './FilterChainAccordion2';
import renderFilterComponent from '../../hooks/renderFilterComponent';

function FlatSection({ mapRef, title, calculation_endpoint, initialSymbols = [], initialName = '', isMain = false, stateId, setStateId, onSectionStateChange, disableAutoChaining = false }) {
  const {
    filterChains,
    setFilterChains,
    initialLoadingMainAccordion,
    mainFiltersAccordionExpanded,
    handleMainFiltersAccordionToggle,
    handleChainAccordionToggle,
    handleFilterValueChange,
    handleAddFilterChain,
  } = useFilterChains(initialSymbols, initialName, disableAutoChaining);

  const [childrenState, setChildrenState] = useState({});

  const handleAccordionStateChange = React.useCallback((chainId, state) => {
    setChildrenState(prevState => {
       if (JSON.stringify(prevState[chainId]) === JSON.stringify(state)) {
        return prevState;
      }
      return {
        ...prevState,
        [chainId]: state
      }
    });
  }, []);

  useEffect(() => {
    if (onSectionStateChange) {
      onSectionStateChange(title, childrenState);
    }
  }, [childrenState, onSectionStateChange, title]);


  return (
    <>
        {filterChains.map((chain, chainIndex) => (
          <FilterChainAccordion2
            key={chain.id}
            chain={chain}
            chainIndex={chainIndex}
            onToggle={handleChainAccordionToggle}
            renderFilterComponent={(chainId, filterSpec) =>
              renderFilterComponent(chainId, filterSpec, handleFilterValueChange)
            }
            mapRef={mapRef}
            onRemove={null}
            calculation_endpoint={calculation_endpoint}
            showDeleteButton={false}
            labelVariant="h8"
            accordionLabel={title}
            staticLabel={true}
            isMain={isMain}
            stateId={stateId}
            setStateId={setStateId}
            onStateChange={handleAccordionStateChange}
          />
        ))}
    </>
  );
}

export default FlatSection;