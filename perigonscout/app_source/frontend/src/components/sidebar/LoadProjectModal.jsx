import React from 'react';
import {
  Modal,
  Box,
  Typography,
  List,
  Button,
  IconButton,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Tooltip,
  Divider,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import CloseIcon from '@mui/icons-material/Close';
import { getDynamicButtonStyle, mainColor, highlightColor, buttonDividerSx, defaultIconColor, neutralGray, defaultBackgroundColor, disabledIconColor } from '../../styles/ButtonStyles';

export const formatDate = (isoString) => {
    if (!isoString) return '';
    const date = new Date(isoString);
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0'); // Month is 0-indexed
    const year = date.getFullYear();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${day}/${month}/${year} ${hours}:${minutes}`;
  };

function LoadProjectModal({
  open,
  onClose,
  savedProjects,
  selectedProjectId,
  onProjectSelect,
  onLoadProject,
  onDeleteProject,
}) {
  const [isConfirmOpen, setIsConfirmOpen] = React.useState(false);

  const handleDeleteClick = () => {
    if (selectedProjectId) {
      setIsConfirmOpen(true);
    }
  };

  const handleConfirmDelete = () => {
    onDeleteProject(selectedProjectId);
    setIsConfirmOpen(false);
  };

  const handleCloseConfirm = () => {
    setIsConfirmOpen(false);
  };

  const selectedProject = savedProjects.find(p => p.id === selectedProjectId);

  return (
    <>
      <Modal
        open={open}
        onClose={onClose}
        aria-labelledby="load-state-modal-title"
        aria-describedby="load-state-modal-description"
      >
        <Box sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 500,
          bgcolor: 'background.paper',
          border: '0px solid #000',
          boxShadow: 24,
          paddingLeft: 3,
          paddingRight: 2,
          py: 3,
          borderRadius: '8px',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <IconButton
            aria-label="close"
            onClick={onClose}
            sx={{
              position: 'absolute',
              right: 16,
              top: 16,
              color: neutralGray,
              '&:hover': { color: defaultIconColor },
            }}
          >
            <CloseIcon sx={{ fontSize: '20px' }} />
          </IconButton>
          <Typography id="load-state-modal-title" variant='caption' color="textSecondary" 
          sx={{ wordBreak: 'break-word', fontWeight: '600', fontSize: '0.875rem', mb: 1, ml: 1.5 }}>
            Wczytaj projekt
          </Typography>
          <List sx={{ 
            overflowY: 'auto', 
            maxHeight: '60vh',
            // Add padding on the right to account for the scrollbar, and on left for alignment
            pr: 1, 
          }}>
            {savedProjects.map((project) => (
              <Button
                key={project.id}
                onClick={() => onProjectSelect(project.id)}
                variant="outlined"
                fullWidth
                sx={{
                  mb: 1.5,
                  p: '11px', // Use consistent padding
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                  textAlign: 'left',
                  textTransform: 'none',
                  borderWidth: '1px', // Always use 2px border
                  borderColor: selectedProjectId === project.id ? disabledIconColor : 'transparent', // Use transparent border for unselected
                  backgroundColor: selectedProjectId === project.id ? 'rgba(167, 167, 167, 0.08)' : 'transparent',
                  '& .MuiTouchRipple-child': {
                    backgroundColor: disabledIconColor,
                  },
                  '&:hover': {
                    borderColor: disabledIconColor,
                    backgroundColor: 'rgba(184, 184, 184, 0.04)',
                    borderWidth: '1px',
                    padding: '11px' // Ensure padding is consistent on hover
                  },
                }}
              >
                <Typography variant="caption" sx={{ fontSize: '0.75rem',fontWeight: '600', color: defaultIconColor }}>{project.name}</Typography>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, fontSize: '0.75rem' }}>
                  {`Created: ${formatDate(project.creation_date)} | Modified: ${formatDate(project.last_edition_date)}`}
                </Typography>
                
                {project.filters_summary && project.filters_summary.length > 0 && (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                    {project.filters_summary.map((filterName, index) => (
                      <Box
                        key={index}
                        sx={{
                          backgroundColor: mainColor,
                          border: '1px solid rgba(0, 0, 0, 0.12)',
                          borderRadius: '4px',
                          px: 0.75,
                          py: 0.0,
                          fontSize: '0.6rem'
                        }}
                      >
                        <Typography variant="caption" color="white" sx={{fontSize: '0.6rem'}}>
                          {filterName}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                )}
              </Button>
            ))}
          </List>
          <Box sx={{ mt: 2, ml: 1.5, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Tooltip title="usuń projekt">
            <IconButton 
              onClick={handleDeleteClick} 
              disabled={!selectedProjectId}
              sx={getDynamicButtonStyle({ disabled: !selectedProjectId })}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
            </Tooltip>
            <Box sx={{ display: 'flex', justifyContent: 'right', gap: 1, mr: 1 }}>
              <Button
                onClick={onClose}
                sx={{
                  color: 'text.secondary',
                  borderRadius: '4px',
                  border: '1px solid transparent',
                  '&:hover': {
                    backgroundColor: defaultBackgroundColor,
                    border: `1px solid ${disabledIconColor}`,
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
                onClick={onLoadProject}
                disabled={!selectedProjectId}
                sx={{
                  backgroundColor: mainColor,
                  color: 'white',
                  borderRadius: '4px',
                  border: `0px solid ${mainColor}`,
                  '&:hover': {
                    backgroundColor: highlightColor,
                  },
                  '&.Mui-disabled': {
                    backgroundColor: 'rgba(0, 0, 0, 0.12)',
                    color: 'rgba(0, 0, 0, 0.26)',
                  },
                  minWidth: '80px',
                  fontSize: '0.875rem',
                  textTransform: 'none',
                  height: '28px'
                }}
              >
                wczytaj
              </Button>
            </Box>
          </Box>
        </Box>
      </Modal>
      <Dialog
        open={isConfirmOpen}
        onClose={handleCloseConfirm}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">{"Confirm Deletion"}</DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            Are you sure you want to delete the project "{selectedProject?.name}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseConfirm}>Cancel</Button>
          <Button onClick={handleConfirmDelete} color="error" autoFocus>
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

export default LoadProjectModal;