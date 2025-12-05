// src/components/sidebar/AddNewFilterChainButton.jsx
import React from 'react';
import { Button, Box, Tooltip, IconButton } from '@mui/material';
import PlaylistAddIcon from '@mui/icons-material/PlaylistAdd';
import { getDynamicButtonStyle } from '../../styles/ButtonStyles';

/**
 * Renders the main 'Add New Filter Chain' button.
 * @param {object} props - Component props.
 * @param {function} props.onClick - Function to call when the button is clicked.
 */
function AddNewFilterChainButton({ onClick, caption }) {
  return (
    <Box sx={{ p: 1.5, display: 'flex', justifyContent: 'center' }}>
      <Tooltip title={caption}>
        <IconButton
        onClick={onClick}
        sx={getDynamicButtonStyle({ disabled: false, isMainButton: false })}
      >
        <PlaylistAddIcon />
        </IconButton>
      </Tooltip>
    </Box>
  );
}

export default AddNewFilterChainButton;