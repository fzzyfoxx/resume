import React from 'react';
import { Box, IconButton, Tooltip, Divider } from '@mui/material';
import HistoryIcon from '@mui/icons-material/History';
import DeleteIcon from '@mui/icons-material/Delete';
import AddFilterButton from './AddFilterButton';
import { getDynamicButtonStyle, buttonDividerSx } from '../../styles/ButtonStyles';


const iconSize = 20;

const FilterChainButtons = ({
  hasChanges,
  handleRestoreValues,
  onRemove,
  chainId,
  marker,
  showDeleteButton = true,
  handleAddOrUpdate,
  handleStop,
  filters,
  status,
  isMain,
  isActual,
  stateId,
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
            <DeleteIcon sx={{ fontSize: iconSize }} />
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
            <HistoryIcon  sx={{ fontSize: iconSize }} />
          </IconButton>
        </Tooltip>
        <Divider orientation="vertical" flexItem sx={buttonDividerSx} />
      <AddFilterButton
        filters={filters}
        status={status}
        hasChanges={hasChanges}
        isMain={isMain}
        isActual={isActual}
        stateId={stateId}
        handleAddOrUpdate={handleAddOrUpdate}
        handleStop={handleStop}
      />
    </Box>
  );
};

export default React.memo(FilterChainButtons);