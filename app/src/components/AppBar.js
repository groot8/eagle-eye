import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';

import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';
import { SwipeableDrawer } from '@material-ui/core';



const useStyles = makeStyles(theme => ({
    root: {
    flexGrow: 1,
    width:'100%',
    
  },
  bar:{
    backgroundColor:'#009BE5'
},
  menuButton: {
    marginRight: theme.spacing(2),
    fontWeight:700,
},
big:{
    width: 30,
    height: 30,

},
  title: {
    flexGrow: 1,
    fontWeight:900
  },
}));

export default function ButtonAppBar() {

  const classes = useStyles();

  return (
    <div className={classes.root}>
      <AppBar position="static" className={classes.bar}>
        <Toolbar>
          <IconButton edge="start" className={classes.menuButton} color="inherit" aria-label="Menu">
            <MenuIcon  className={classes.big}/>
          </IconButton>
          <Typography variant="h3" className={classes.title}>
            EAGLE EYE
          </Typography>
          
        </Toolbar>
      </AppBar>
      <SwipeableDrawer></SwipeableDrawer>
    </div>
  );
}