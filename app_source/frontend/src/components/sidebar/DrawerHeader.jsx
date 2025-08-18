// src/components/sidebar/DrawerHeader.jsx
import React from 'react';
import { IconButton } from '@mui/material';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import { styled, useTheme } from '@mui/material/styles';

const StyledDrawerHeader = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(0, 1),
  ...theme.mixins.toolbar,
  justifyContent: 'flex-end',
}));

/**
 * Renders the header for the sidebar drawer.
 * @param {object} props - Component props.
 * @param {function} props.handleDrawerClose - Function to call when the drawer close button is clicked.
 */
function DrawerHeader({ handleDrawerClose }) {
  const theme = useTheme();
  return (
    <StyledDrawerHeader>
      <IconButton onClick={handleDrawerClose}>
        {theme.direction === 'ltr' ? <ChevronLeftIcon /> : <ChevronRightIcon />}
      </IconButton>
    </StyledDrawerHeader>
  );
}

export default DrawerHeader;