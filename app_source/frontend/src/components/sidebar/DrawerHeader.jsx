import React from 'react';
import { IconButton, Box, Tooltip, Divider } from '@mui/material';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import SaveIcon from '@mui/icons-material/Save';
import FileOpenIcon from '@mui/icons-material/FileOpen';
import { styled } from '@mui/material/styles';
import SaveAsIcon from '@mui/icons-material/SaveAs';
import NoteAddIcon from '@mui/icons-material/NoteAdd';
import { getDynamicButtonStyle, buttonDividerSx } from '../../styles/ButtonStyles';

const StyledDrawerHeader = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(0, 1),
  minHeight: '36px',
  height: '36px',
  justifyContent: 'space-between', // Changed to space-between
}));

const iconSize = 20;

/**
 * Renders the header for the sidebar drawer.
 * @param {object} props - Component props.
 * @param {function} props.handleDrawerClose - Function to call when the drawer close button is clicked.
 * @param {function} props.onSaveState - Function to save the state.
 * @param {function} props.onLoadState - Function to load the state.
 * @param {function} props.onSaveAs - Function to handle "Save As" action.
 * @param {function} props.onNewProject - Function to handle "New Project" action. // Add onNewProject prop
 */
function DrawerHeader({ handleDrawerClose, onSaveState, onLoadState, onSaveAs, onNewProject }) { // Add onNewProject to props
  return (
    <StyledDrawerHeader>
      <Box
      sx={{
        width: '90%',
        maxWidth: 'calc(100% - 53px)',
        display: 'flex',
        justifyContent: 'left',
        gap: 2,
        my: 1,
        ml: 2.0,
        mr: 2.0,
      }}
      >
        <Tooltip title="nowy projekt"> {/* Add New Project button */}
          <IconButton onClick={onNewProject} sx={getDynamicButtonStyle({disabled: false, isMainButton: false})}>
            <NoteAddIcon  sx={{ fontSize: iconSize }} />
          </IconButton>
        </Tooltip>
        <Divider orientation="vertical" flexItem sx={buttonDividerSx} />
        <Tooltip title="zapisz">
          <IconButton onClick={onSaveState} sx={getDynamicButtonStyle({disabled: false, isMainButton: false})}>
            <SaveIcon  sx={{ fontSize: iconSize }} />
          </IconButton>
        </Tooltip>
        <Divider orientation="vertical" flexItem sx={buttonDividerSx} />
        <Tooltip title="zapisz jako"> {/* Add Save As button */}
          <IconButton onClick={onSaveAs} sx={getDynamicButtonStyle({disabled: false, isMainButton: false})}>
            <SaveAsIcon  sx={{ fontSize: iconSize }} />
          </IconButton>
        </Tooltip>
        <Divider orientation="vertical" flexItem sx={buttonDividerSx} />
        <Tooltip title="wczytaj projekt">
          <IconButton onClick={onLoadState} sx={getDynamicButtonStyle({disabled: false, isMainButton: false})}>
            <FileOpenIcon  sx={{ fontSize: iconSize }} />
          </IconButton>
        </Tooltip>
      </Box>
        <IconButton
          onClick={handleDrawerClose}
          sx={{
            paddingRight: '0px',
          }}
        >
          <ChevronLeftIcon />
        </IconButton>
    </StyledDrawerHeader>
  );
}

export default DrawerHeader;