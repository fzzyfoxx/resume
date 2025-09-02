import React from 'react';
import { IconButton, Tooltip } from '@mui/material';
import { useFilterQuery } from '../../hooks/useFilterQuery';
import CheckIcon from '@mui/icons-material/Check';
import RefreshIcon from '@mui/icons-material/Refresh';
import StopIcon from '@mui/icons-material/Stop';
import EditIcon from '@mui/icons-material/Edit';

const AccordionStatusButton = ({
  indicator,
}) => {

  const handleClick = (event) => {
    event.stopPropagation();
    if (!indicator.active) {
      return;
    }

    switch (indicator.status) {
      case 'warning':
        //handleAddOrUpdate();
        break;
      case 'loading':
        //handleStop();
        break;
      default:
        // No action for 'ok' or 'default'
        break;
    }
  };

  const iconSize = 14;

  const getIcon = () => {
    switch (indicator.status) {
      case 'ok':
        return <CheckIcon sx={{ fontSize: iconSize }} />;
      case 'warning':
        return <RefreshIcon sx={{ fontSize: iconSize }} />;
      case 'loading':
        return <StopIcon sx={{ fontSize: iconSize }} />;
      case 'default':
        return <EditIcon sx={{ fontSize: iconSize }} />;
      default:
        return null;
    }
  };

  return (
    <Tooltip title={indicator.caption}>
      <span style={{ display: 'flex', alignItems: 'center' }}>
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
                  backgroundColor: 'transparent',
                  color: 'rgb(88, 88, 88)',
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
      </span>
    </Tooltip>
  );
};

export default AccordionStatusButton;