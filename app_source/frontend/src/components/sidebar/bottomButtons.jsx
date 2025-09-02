import React from 'react';
import { Box, IconButton, Tooltip, Divider } from '@mui/material';
import HistoryIcon from '@mui/icons-material/History';
import DeleteIcon from '@mui/icons-material/Delete';
import AddFilterButton from './AddFilterButton';
import { getDynamicButtonStyle, buttonDividerSx } from '../../styles/ButtonStyles';




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
  isActual,
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
      {showDeleteButton && (
        <>
        <Tooltip title="usuń filtr">
          <IconButton
            onClick={() => onRemove(chainId, marker)}
            sx={getDynamicButtonStyle({disabled: false, isMainButton: false})}
          >
            <DeleteIcon />
          </IconButton>
        </Tooltip>
        <Divider orientation="vertical" flexItem sx={buttonDividerSx} />
        </>
      )}
        <Tooltip title={hasChanges ? "cofnij zmiany" : "brak zmian do cofnięcia"}>
          <IconButton
            onClick={handleRestoreValues}
            sx={getDynamicButtonStyle({disabled: !hasChanges, isMainButton: false})}
          >
            <HistoryIcon />
          </IconButton>
        </Tooltip>
        <Divider orientation="vertical" flexItem sx={buttonDividerSx} />

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