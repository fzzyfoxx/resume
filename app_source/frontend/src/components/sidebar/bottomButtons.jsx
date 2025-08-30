import React from 'react';
import { Box, IconButton, Tooltip } from '@mui/material';
import HistoryIcon from '@mui/icons-material/History';
import DeleteIcon from '@mui/icons-material/Delete';
import AddFilterButton from './AddFilterButton';

const FilterChainButtons = ({
  hasChanges,
  handleRestoreValues,
  onRemove,
  chainId,
  marker,
  filters,
  addFilterStatus,
  setAddFilterStatus,
  setImplied,
  mapRef,
  accordionSummary,
  setMarker,
  memoizedHasChanges,
  calculation_endpoint,
  showDeleteButton = true,
  isMain,
  filterStateId,
  setFilterStateId,
  stateId,
  setStoredStateId,
  isActual
}) => {
  return (
    <Box
      sx={{
        width: '90%',
        maxWidth: 'calc(100% - 53px)',
        display: 'flex',
        justifyContent: 'right',
        gap: 2,
        my: 1,
        ml: 2.0,
        mr: 2.0,
      }}
    >
      {hasChanges && (
        <Tooltip title="Cofnij zmiany">
          <IconButton
            onClick={handleRestoreValues}
            sx={{
              backgroundColor: 'gray',
              '&:hover': {
                backgroundColor: 'darkgray',
              },
              color: 'white',
              borderRadius: '50%',
              width: 32,
              height: 32,
            }}
          >
            <HistoryIcon />
          </IconButton>
        </Tooltip>
      )}

      {showDeleteButton && ( // Conditionally render the delete button
        <Tooltip title="Usuń filtr">
          <IconButton
            onClick={() => onRemove(chainId, marker)}
            sx={{
              backgroundColor: 'gray',
              '&:hover': {
                backgroundColor: 'darkgray',
              },
              color: 'white',
              borderRadius: '50%',
              width: 32,
              height: 32,
            }}
          >
            <DeleteIcon />
          </IconButton>
        </Tooltip>
      )}

      <AddFilterButton
        filters={filters}
        status={addFilterStatus}
        onStatusChange={setAddFilterStatus}
        onImpliedChange={setImplied}
        mapRef={mapRef}
        accordionSummary={accordionSummary}
        marker={marker}
        setMarker={setMarker}
        hasChanges={memoizedHasChanges}
        endpoint={calculation_endpoint}
        isMain={isMain}
        filterStateId={filterStateId}
        setFilterStateId={setFilterStateId}
        stateId={stateId}
        setStoredStateId={setStoredStateId}
        isActual={isActual}
      />
    </Box>
  );
};

export default FilterChainButtons;