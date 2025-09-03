import React, { useState, useEffect, useRef } from 'react';
import { Box, Typography, Input, IconButton } from '@mui/material';
import CircularLoader from '../common/CircularLoader';
import HideLayer from '../../drawing/HideLayer';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useDebounce } from '../../hooks/useDebounce';

const AccordionSummaryContent = ({
  summaryParts,
  chain,
  marker,
  mapRef,
  variant,
  isStatic = false,
  title,
  setTitle,
  onToggle,
  isExpanded,
  statusButton,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [localTitle, setLocalTitle] = useState(title); // Local state for the input
  const debouncedTitle = useDebounce(localTitle, 500); // Debounce the local title (500ms delay)
  const inputRef = useRef(null);

  // Effect to propagate the debounced title change upwards
  useEffect(() => {
    if (debouncedTitle !== title) {
      setTitle(debouncedTitle);
    }
  }, [debouncedTitle, setTitle, title]);

  // When the parent title changes (e.g., on load), update the local title
  useEffect(() => {
    setLocalTitle(title);
  }, [title]);

  const handleDoubleClick = (event) => {
    if (!isStatic) {
      event.stopPropagation();
      setIsEditing(true);
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.select(); // Select the text in the Input
        }
      }, 0);
    }
  };

  const handleTitleChange = (event) => {
    setLocalTitle(event.target.value);
  };

  const handleTitleBlur = () => {
    setIsEditing(false);
    // Here you could add a function to persist the new title if needed
  };

  const handleInputKeyDown = (event) => {
    if (event.key === 'Enter') {
      handleTitleBlur();
    }
  };

  return (
    <Box sx={{ width: '100%', ml: 1.5, py: 0.5, mb: 0.75 }}>
      {/* Top Box: Title and Buttons */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        {/* Title Box */}
        <Box
          onDoubleClick={handleDoubleClick}
          onClick={(e) => e.stopPropagation()} // Prevent accordion toggle on click
          onFocus={(e) => e.stopPropagation()}
          sx={{ flexGrow: 1, cursor: isStatic ? 'default' : 'pointer', backgroundColor: 'transparent' }}
        >
          {isEditing ? (
            <Input
            inputRef={inputRef} // Attach the ref to the Input
            value={localTitle}
            onChange={handleTitleChange}
            onBlur={handleTitleBlur}
            onKeyDown={handleInputKeyDown}
            autoFocus
            disableUnderline // Disable the default underline behavior
            sx={{
              fontSize: variant === 'caption' ? '0.75rem' : '1rem',
              backgroundColor: 'transparent', // Always transparent background
              borderBottom: '1px solid gray', // Gray bottom line during editing
              outline: 'none', // Remove default focus outline
              minWidth: '10px',
              width: `${Math.max(title.length, 10)}ch`,
                '&:hover:not(.Mui-disabled):before': {
                  borderBottom: '1px solid gray',
                },
                '&:before, &:after': {
                  display: 'none', // Hide default pseudo-elements
                },
              }}
            />
          ) : (
            <Typography variant={variant} color="textSecondary" sx={{ wordBreak: 'break-word', fontSize: variant === 'caption' ? '0.875rem' : '1rem' }}>
              {title}
            </Typography>
          )}
        </Box>

        {/* Buttons Box */}
        <Box sx={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
          {statusButton}
          {marker && <HideLayer marker={marker} mapRef={mapRef} />}
          {onToggle && (
            <IconButton onClick={onToggle} size="small" sx={{ ml: 0.5 }} disableRipple>
              <ExpandMoreIcon
                sx={{
                  transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s',
                }}
              />
            </IconButton>
          )}
        </Box>

      </Box>

      {/* Bottom Box: Summary Chain */}
      {!isStatic && summaryParts.length > 0 && (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.3, mt: 0.3, pr: '32px' }}>
          {summaryParts.map((part, index) => (
            <Box
              key={index}
              sx={{
                backgroundColor: 'rgba(0, 0, 0, 0.08)',
                border: '1px solid rgba(0, 0, 0, 0.12)',
                borderRadius: '4px',
                px: 0.75,
                py: 0.0,
                fontSize: '0.6rem'
              }}
            >
              <Typography variant="caption" color="textSecondary" sx={{fontSize: '0.6rem',}}>
                {part}
              </Typography>
            </Box>
          ))}
        </Box>
      )}

      {chain.isLoading && <CircularLoader size={14} sx={{ mt: 0.5 }} />}
    </Box>
  );
};

export default AccordionSummaryContent;