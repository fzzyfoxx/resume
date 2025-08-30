import React from 'react';
import { IconButton, Box, Tooltip } from '@mui/material';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import SaveIcon from '@mui/icons-material/Save';
import RestoreIcon from '@mui/icons-material/Restore';
import { styled } from '@mui/material/styles';

const StyledDrawerHeader = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(0, 1),
  minHeight: '36px',
  height: '36px',
  justifyContent: 'space-between', // Changed to space-between
}));

/**
 * Renders the header for the sidebar drawer.
 * @param {object} props - Component props.
 * @param {function} props.handleDrawerClose - Function to call when the drawer close button is clicked.
 * @param {function} props.onSaveState - Function to save the state.
 * @param {function} props.onLoadState - Function to load the state.
 */
function DrawerHeader({ handleDrawerClose, onSaveState, onLoadState }) {
  return (
    <StyledDrawerHeader>
      <Box>
        <Tooltip title="Save Current State">
          <IconButton onClick={onSaveState}>
            <SaveIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Load Last State">
          <IconButton onClick={onLoadState}>
            <RestoreIcon />
          </IconButton>
        </Tooltip>
      </Box>
      <Tooltip title="Hide Panel">
        <IconButton
          onClick={handleDrawerClose}
          sx={{
            paddingRight: '0px',
          }}
        >
          <ChevronLeftIcon />
        </IconButton>
      </Tooltip>
    </StyledDrawerHeader>
  );
}

export default DrawerHeader;