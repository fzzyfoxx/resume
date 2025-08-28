import React from 'react';
import { Box, Typography } from '@mui/material';
import CircularLoader from '../common/CircularLoader';
import HideLayer from '../../drawing/HideLayer';

const AccordionSummaryContent = ({ accordionTitle, chain, marker, mapRef, variant, label, static: isStatic = false }) => {
  return (
    <>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          flexGrow: 1,
          alignItems: 'flex-start',
          ml: 1.5,
          py: 0.5,
        }}
      >
        <Typography
          variant={variant}
          color="textSecondary"
          sx={{
            wordBreak: 'break-word',
          }}
        >
          {isStatic ? label : accordionTitle} {/* Use label if static, otherwise accordionTitle */}
        </Typography>
        {chain.isLoading && (
          <CircularLoader size={14} sx={{ mt: 0.5, flexShrink: 0 }} />
        )}
      </Box>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
        }}
      >
        {marker && <HideLayer marker={marker} mapRef={mapRef} />}
      </Box>
    </>
  );
};

export default AccordionSummaryContent;