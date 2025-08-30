import React, { useState, useEffect } from 'react';
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

function FiltersSection({ mapRef, title, calculation_endpoint, initialSymbols = [], initialName = '', isMain = false, stateId, setStateId, onSectionStateChange, disableAutoChaining = false }) {
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

  const [dialogOpen, setDialogOpen] = useState(false);
  const [chainToRemove, setChainToRemove] = useState(null);
  const [markerToRemove, setMarkerToRemove] = useState(null);
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

    setFilterChains((prevChains) =>
      prevChains.filter((chain) => chain.id !== chainToRemove)
    );
    handleCloseDialog();
  };

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
            renderFilterComponent={(chainId, filterSpec) =>
              renderFilterComponent(chainId, filterSpec, handleFilterValueChange)
            }
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
          />
        ))}

        <AddNewFilterChainButton 
          onClick={handleAddFilterChain}
          caption="Dodaj kolejny filtr"
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