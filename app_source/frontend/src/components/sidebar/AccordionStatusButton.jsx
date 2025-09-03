import React, { useCallback } from 'react';
import { IconButton, Tooltip, CircularProgress, Box  } from '@mui/material';
import { useFilterQuery } from '../../hooks/useFilterQuery';
import CheckIcon from '@mui/icons-material/Check';
import RefreshIcon from '@mui/icons-material/Refresh';
import StopIcon from '@mui/icons-material/Stop';
import EditIcon from '@mui/icons-material/Edit';
import { smallIconSize, defaultIconColor, disabledIconColor } from '../../styles/ButtonStyles';

const AccordionStatusButton = ({
  indicator,
  handleAddOrUpdate,
  handleStop,
  isExpanded
}) => {

  const handleClick = useCallback((event) => {
    event.stopPropagation();
    if (!indicator.active) {
      return;
    }

    switch (indicator.status) {
      case 'warning':
        handleAddOrUpdate();
        break; // Added break for correctness
      case 'loading':
        handleStop();
        break; // Added break for correctness
      default:
        // No action for 'ok' or 'default'
        break;
    }
  }, [indicator.active, indicator.status, handleAddOrUpdate, handleStop]);

  const getIcon = () => {
    switch (indicator.status) {
      case 'ok':
        return <CheckIcon sx={{ fontSize: smallIconSize }} />;
      case 'warning':
        return <RefreshIcon sx={{ fontSize: smallIconSize }} />;
      case 'loading':
        return <StopIcon sx={{ fontSize: smallIconSize }} />;
      case 'default':
        return <EditIcon sx={{ fontSize: smallIconSize }} />;
      default:
        return null;
    }
  };

  return (
    <Tooltip title={indicator.caption}>
      <Box sx={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
        <IconButton
          onClick={handleClick}
          size='small'
          disabled={!indicator.active}
          sx={{
            width: 20,
            height: 20,
            p: 0.0,
            ml: 0.5,
            backgroundColor: indicator.color,
            color: 'white',
            '&:hover': indicator.active
              ? {
                  backgroundColor: indicator.color,
                  color: 'white',
                  border: `1px solid rgb(88, 88, 88)`,
                }
              : {},
            '&.Mui-disabled': {
              backgroundColor: indicator.color,
              color: 'white',
              opacity: 0.7,
            },
          }}
        >
          {getIcon()}
        </IconButton>
        {indicator.status === 'loading' && !isExpanded && (
          <CircularProgress
            size={26}
            sx={{
              color: defaultIconColor,
              position: 'absolute',
              top: -3,
              left: 1,
              zIndex: 1,
            }}
          />
        )}
      </Box>
    </Tooltip>
  );
};

export default AccordionStatusButton;