// src/components/sidebar/AddNewFilterChainButton.jsx
import React from 'react';
import { Button, Box } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';

/**
 * Renders the main 'Add New Filter Chain' button.
 * @param {object} props - Component props.
 * @param {function} props.onClick - Function to call when the button is clicked.
 */
function AddNewFilterChainButton({ onClick, caption }) {
  return (
    <Box sx={{ p: 1.5, display: 'flex', justifyContent: 'center' }}>
      <Button
        variant="outlined"
        startIcon={<AddIcon />}
        onClick={onClick}
        sx={{ 
          fontSize: '0.85rem', 
          padding: '6px 14px',
          backgroundColor: 'gray',
          borderColor: 'darkgray',
          '&:hover': { backgroundColor: 'darkgray', borderColor: 'darkgray' },
          color: 'white',
        }}
      >
        {caption}
      </Button>
    </Box>
  );
}

export default AddNewFilterChainButton;