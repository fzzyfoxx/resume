import React from 'react';
import {
  Modal,
  Box,
  Typography,
  List,
  Button,
} from '@mui/material';

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
}) {
  return (
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
        border: '2px solid #000',
        boxShadow: 24,
        p: 4,
        display: 'flex',
        flexDirection: 'column'
      }}>
        <Typography id="load-state-modal-title" variant="h6" component="h2" sx={{ mb: 2 }}>
          Load Project
        </Typography>
        <List sx={{ overflowY: 'auto', maxHeight: '60vh', pr: 1 }}>
          {savedProjects.map((project) => (
            <Button
              key={project.id}
              onClick={() => onProjectSelect(project.id)}
              variant={selectedProjectId === project.id ? 'contained' : 'outlined'}
              fullWidth
              sx={{
                mb: 1.5,
                p: 1.5,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-start',
                textAlign: 'left',
                textTransform: 'none'
              }}
            >
              <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>{project.name}</Typography>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                Created: {formatDate(project.creation_date)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Modified: {formatDate(project.last_edition_date)}
              </Typography>
              
              {project.filters_summary && project.filters_summary.length > 0 && (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                  {project.filters_summary.map((filterName, index) => (
                    <Box
                      key={index}
                      sx={{
                        backgroundColor: 'rgba(0, 0, 0, 0.08)',
                        border: '1px solid rgba(0, 0, 0, 0.12)',
                        borderRadius: '4px',
                        px: 0.75,
                        py: 0.0,
                      }}
                    >
                      <Typography variant="caption" color="textSecondary" sx={{fontSize: '0.6rem'}}>
                        {filterName}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              )}
            </Button>
          ))}
        </List>
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
          <Button onClick={onClose}>Cancel</Button>
          <Button onClick={onLoadProject} disabled={!selectedProjectId}>
            Load
          </Button>
        </Box>
      </Box>
    </Modal>
  );
}

export default LoadProjectModal;