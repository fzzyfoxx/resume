import React, { useState, useEffect, useRef, useCallback, useMemo  } from 'react';
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
import { getIndicatorColor } from '../../utils/filterChainUtils';

function FiltersSection({ 
  mapRef, 
  title, 
  calculation_endpoint, 
  initialSymbols = [], 
  initialName = '', 
  isMain = false, 
  stateId, 
  setStateId, 
  onSectionStateChange, 
  disableAutoChaining = false, 
  loadedState, 
  fitBounds,
  onMarkersChange,
  allFilterStateIds,
  sourceStateId,
  setSourceStateId,
  isTarget = false
  }) {
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
  const [bulkUpdate, setBulkUpdate] = useState(0);
  const [bulkStop, setBulkStop] = useState(0);
  const bulkCountersRef = useRef({ update: 0, stop: 0 });
  const [areAllVisible, setAreAllVisible] = useState(true);
  const [chainVisibility, setChainVisibility] = useState({}); // per-chain visibility

  const allMarkers = useMemo(() => Object.values(chainMarkers).filter(Boolean), [chainMarkers]);

  const handleToggleAllLayers = (isVisible) => {
    if (!mapRef.current) return;
    allMarkers.forEach(marker => {
      if (isVisible) {
        if (!mapRef.current.hasLayer(marker)) mapRef.current.addLayer(marker);
      } else {
        if (mapRef.current.hasLayer(marker)) mapRef.current.removeLayer(marker);
      }
    });
    setAreAllVisible(isVisible);
    // Sync all child visibility states
    setChainVisibility(prev => {
      const updated = { ...prev };
      filterChains.forEach(c => { updated[c.id] = isVisible; });
      return updated;
    });
  };

  const handleSingleLayerVisibility = useCallback((chainId, visible, marker) => {
    // Toggle the specific marker(s) on map (HideLayer already handles actual layer ops)
    setChainVisibility(prev => {
      const updated = { ...prev, [chainId]: visible };
      // Update global flag if all true or any false
      const values = Object.values(updated);
      if (values.length > 0) {
        setAreAllVisible(values.every(v => v === true));
      } else {
        setAreAllVisible(true);
      }
      return updated;
    });
  }, []);

  const mainIndicator = useMemo(() => {
    const childStateValues = Object.values(childrenState);
    if (childStateValues.length === 0) {
      return getIndicatorColor(false, 'add', true);
    }

    const visualStatuses = childStateValues.map(c => c.visualStatus || 'default');

    let mainVisual;
    if (visualStatuses.some(s => s === 'loading')) {
      mainVisual = 'loading';
    } else if (visualStatuses.some(s => s === 'warning')) {
      mainVisual = 'warning';
    } else if (visualStatuses.every(s => s === 'default')) {
      mainVisual = 'default';
    } else if (visualStatuses.some(s => s === 'ok')) {
      mainVisual = 'ok';
    } else {
      mainVisual = 'default';
    }

    switch (mainVisual) {
      case 'loading':
        return getIndicatorColor(false, 'stop', true);
      case 'warning':
        return getIndicatorColor(true, 'update', true);
      case 'ok':
        return getIndicatorColor(false, 'update', true);
      case 'default':
      default:
        return getIndicatorColor(false, 'add', true);
    }
  }, [childrenState]);

  const handleMainAddOrUpdate = useCallback(() => {
    bulkCountersRef.current.update += 1;
    setBulkUpdate({ id: bulkCountersRef.current.update });
  }, []);

  const handleMainStop = useCallback(() => {
    bulkCountersRef.current.stop += 1;
    const id = bulkCountersRef.current.stop;
    console.log(`[FiltersSection] Main STOP clicked -> bulkStop id=${id}`);
    setBulkStop({ id });
  }, []);

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
    setChainVisibility(prev => (
      prev.hasOwnProperty(chainId) ? prev : { ...prev, [chainId]: true }
    ));
  }, []);

  const handleAccordionStateChange = useCallback((chainId, state) => {
        // LOG: Check if the parent is receiving the state change
        console.log(`[FiltersSection] Received state update from child: ${chainId}`, state);
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

    setChainVisibility(prev => {
      const nv = { ...prev };
      delete nv[chainToRemove];
      return nv;
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
        mainIndicator={mainIndicator}
        onMainAddOrUpdate={handleMainAddOrUpdate}
        onMainStop={handleMainStop}
        allMarkers={allMarkers}
        mapRef={mapRef}
        areAllVisible={areAllVisible}
        onToggleAllLayers={handleToggleAllLayers}
        hasChildren={filterChains.length > 0}
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
            allFilterStateIds={allFilterStateIds}
            sourceStateId={sourceStateId}
            setSourceStateId={setSourceStateId}
            isTarget={isTarget}
            onBulkUpdate={bulkUpdate}
            onBulkStop={bulkStop}
            isLayerVisible={chainVisibility[chain.id] ?? true}
            setLayerVisible={(v) => handleSingleLayerVisibility(chain.id, v, chainMarkers[chain.id])}
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