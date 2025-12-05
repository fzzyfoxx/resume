// src/components/sidebar/AddButton.jsx
import React from 'react';
import { Button, Box } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';

/**
 * Renders an 'ADD' button for a completed filter chain.
 * @param {object} props - Component props.
 * @param {function} props.onClick - Function to call when the button is clicked.
 */
function AddButton({ onClick }) {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'flex-end', pr: 2, pb: 1 }}>
      <Button
        variant="outlined"
        startIcon={<AddIcon />}
        onClick={onClick}
        sx={{
            fontSize: '0.85rem', // Matched 'ADD NEW FILTER CHAIN' button
            padding: '6px 14px', // Matched 'ADD NEW FILTER CHAIN' button
            minWidth: 'unset'
        }}
      >
        ADD
      </Button>
    </Box>
  );
}

export default AddButton;