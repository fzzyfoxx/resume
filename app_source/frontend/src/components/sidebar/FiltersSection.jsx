import React, { useState, useEffect, useRef, useCallback  } from 'react';
import {
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
} from '@mui/material';

import useFilterChains from '../../hooks/useFilterChains';
import MainFilterAccordion from './MainFilterAccordion';
import FilterChainAccordion2 from './FilterChainAccordion2';
import AddNewFilterChainButton from './AddNewFilterChainButton';
import renderFilterComponent from '../../hooks/renderFilterComponent';

function FiltersSection({ mapRef, title, calculation_endpoint, initialSymbols = [], initialName = '', isMain = false, stateId, setStateId, onSectionStateChange, disableAutoChaining = false, loadedState, fitBounds, onMarkersChange }) {
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

  const [dialogOpen, setDialogOpen] = useState(false);
  const [chainToRemove, setChainToRemove] = useState(null);
  const [markerToRemove, setMarkerToRemove] = useState(null);
  const [childrenState, setChildrenState] = useState({});
  const [chainMarkers, setChainMarkers] = useState({});
  const loadedStateRef = useRef(null);

  useEffect(() => {
    if (onMarkersChange) {
      onMarkersChange(title, chainMarkers);
    }
  }, [chainMarkers, title, onMarkersChange]);


  useEffect(() => {
    // Only proceed if loadedState has changed and is different from what we've already processed.
    if (loadedState && loadedStateRef.current !== loadedState) {
      setChainMarkers({}); // Reset the local marker tracking state
      setChildrenState({}); // Reset state when a new state is loaded
      loadedStateRef.current = loadedState; // Mark as processed
      const isEffectivelyEmpty = !Object.values(loadedState).some(
        chain => chain.storedFilterValues && chain.storedFilterValues.length > 0
      );

      if (!isEffectivelyEmpty) {
        restoreChainsFromState(loadedState);
      } else {
        // If loaded state for filters is empty or has no values, clear existing chains
        setFilterChains([]);
      }
    } else if (loadedState === null && loadedStateRef.current !== null) {
      // Handle "New Project" case where loadedState becomes null
      setFilterChains([]);
      setChainMarkers({});
      setChildrenState({});
      loadedStateRef.current = null;
    }
  }, [loadedState, restoreChainsFromState, setFilterChains]);

  const handleMarkerCreated = useCallback((chainId, marker) => {
    setChainMarkers(prevMarkers => ({
      ...prevMarkers,
      [chainId]: marker
    }));
  }, []);

  const handleAccordionStateChange = useCallback((chainId, state) => {
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

  const handleOpenDialog = (chainId, marker) => {
    setChainToRemove(chainId);
    setMarkerToRemove(marker);
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
    setChainToRemove(null);
  };

  const handleRemoveFilterChain = () => {
    if (markerToRemove && mapRef?.current) {
      mapRef.current.removeLayer(markerToRemove);
    }

    setChildrenState(prevState => {
      const newState = { ...prevState };
      delete newState[chainToRemove];
      return newState;
    });

    setChainMarkers(prevMarkers => {
      const newMarkers = { ...prevMarkers };
      delete newMarkers[chainToRemove];
      return newMarkers;
    });

    setFilterChains((prevChains) =>
      prevChains.filter((chain) => chain.id !== chainToRemove)
    );
    handleCloseDialog();
  };

  const memoizedRenderFilterComponent = useCallback((chainId, filterSpec) =>
    renderFilterComponent(chainId, filterSpec, handleFilterValueChange),
    [handleFilterValueChange]
  );

  return (
    <>
      <MainFilterAccordion
        caption={title}
        expanded={mainFiltersAccordionExpanded}
        onToggle={handleMainFiltersAccordionToggle}
        isLoading={initialLoadingMainAccordion}
      >
        {filterChains.map((chain, chainIndex) => (
          <FilterChainAccordion2
            key={chain.id}
            chain={chain}
            chainIndex={chainIndex}
            onToggle={handleChainAccordionToggle}
            renderFilterComponent={memoizedRenderFilterComponent}
            mapRef={mapRef}
            onRemove={handleOpenDialog}
            calculation_endpoint={calculation_endpoint}
            showDeleteButton={true}
            labelVariant="caption"
            accordionLabel={null}
            staticLabel={false}
            isMain={isMain}
            stateId={stateId}
            setStateId={setStateId}
            onStateChange={handleAccordionStateChange}
            onMarkerCreated={handleMarkerCreated}
            fitBounds={fitBounds}
            loadedChainName={chain.chainName} 
          />
        ))}

        <AddNewFilterChainButton 
          onClick={handleAddFilterChain}
          caption="dodaj filtr"
         />
      </MainFilterAccordion>

      {/* Confirmation Dialog */}
      <Dialog open={dialogOpen} onClose={handleCloseDialog}>
        <DialogTitle>Potwierdź usunięcie filtra</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Czy napewno chcesz usunąć ten filtr?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog} color="primary">
            Anuluj
          </Button>
          <Button onClick={handleRemoveFilterChain} color="primary">
            Usuń
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

export default FiltersSection;