import React, { useState, useEffect, useRef } from 'react';

import useFilterChains from '../../hooks/useFilterChains';
import FilterChainAccordion2 from './FilterChainAccordion2';
import renderFilterComponent from '../../hooks/renderFilterComponent';

function FlatSection({ mapRef, title, calculation_endpoint, initialSymbols = [], initialName = '', isMain = false, stateId, setStateId, onSectionStateChange, disableAutoChaining = false, loadedState }) {
  const {
    filterChains,
    setFilterChains,
    initialLoadingMainAccordion,
    mainFiltersAccordionExpanded,
    handleMainFiltersAccordionToggle,
    handleChainAccordionToggle,
    handleFilterValueChange,
    handleAddFilterChain,
    restoreChainsFromState,
  } = useFilterChains(initialSymbols, initialName, disableAutoChaining);

  const [childrenState, setChildrenState] = useState({});
  const loadedStateRef = useRef(null);

  useEffect(() => {
    // Only proceed if loadedState has changed and is different from what we've already processed.
    if (loadedState && loadedStateRef.current !== loadedState) {
      setChildrenState({}); // Reset state when a new state is loaded
      loadedStateRef.current = loadedState; // Mark as processed
      const isEffectivelyEmpty = !Object.values(loadedState).some(
        chain => chain.storedFilterValues && chain.storedFilterValues.length > 0
      );

      if (!isEffectivelyEmpty) {
        restoreChainsFromState(loadedState);
      } else {
        // If loaded state for this section is empty, clear any existing chains and then load the initial one.
        setFilterChains([]);
        handleAddFilterChain(true);
      }
    }
  }, [loadedState, restoreChainsFromState, handleAddFilterChain, setFilterChains]);

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