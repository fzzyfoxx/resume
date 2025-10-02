import React from 'react';
import {
  Modal,
  Box,
  Typography,
  TextField,
  Button,
  Divider,
} from '@mui/material';
import { mainColor, highlightColor, buttonDividerSx } from '../../styles/ButtonStyles';

function SaveProjectAsModal({ open, onClose, projectName, onProjectNameChange, onSave }) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      aria-labelledby="save-as-modal-title"
    >
      <Box sx={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: 320,
        bgcolor: 'background.paper',
        border: '0px solid #000',
        boxShadow: 24,
        p: 2,
        borderRadius: '8px',
      }}>
        <Typography id="save-as-modal-title" variant='caption' color="textSecondary" sx={{ wordBreak: 'break-word', fontWeight: '600', fontSize: '0.875rem' }}>
          Zapisz projekt
        </Typography>
        <TextField
          autoFocus
          margin="dense"
          id="name"
          label="nazwa projektu"
          type="text"
          fullWidth
          variant="standard"
          value={projectName}
          onChange={onProjectNameChange}
          sx={{
            mt: 1,
            '& .MuiInputLabel-root': {
              color: 'gray',
              fontSize: '0.875rem'
            },
            '& .MuiInputLabel-root.Mui-focused': {
              color: mainColor,
            },
            '& .MuiInput-underline:before': {
              borderBottomColor: 'gray',
            },
            '& .MuiInput-underline:hover:not(.Mui-disabled):before': {
              borderBottomColor: highlightColor,
            },
            '& .MuiInput-underline:after': {
              borderBottomColor: mainColor,
            },
          }}
          InputProps={{
            style: {
              fontSize: '0.875rem',
              color: 'secondary'
            }
          }}
        />
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'right', gap: 1 }}>
          <Button
            onClick={onClose}
            sx={{
              color: 'text.secondary',
              borderRadius: '4px',
              '&:hover': {
                backgroundColor: 'rgba(0, 0, 0, 0.04)'
              },
              minWidth: '80px',
              fontSize: '0.875rem',
              textTransform: 'none',
              height: '28px'
            }}
          >
            anuluj
          </Button>
          <Divider orientation="vertical" flexItem sx={buttonDividerSx} />
          <Button
            onClick={onSave}
            sx={{
              backgroundColor: mainColor,
              color: 'white',
              borderRadius: '4px',
              border: `0px solid ${mainColor}`,
              '&:hover': {
                backgroundColor: highlightColor,
              },
              minWidth: '80px',
              fontSize: '0.875rem',
              textTransform: 'none',
              height: '28px'
            }}
          >
            zapisz
          </Button>
        </Box>
      </Box>
    </Modal>
  );
}

export default SaveProjectAsModal;