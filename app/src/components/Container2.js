import React from 'react';
import CssBaseline from '@material-ui/core/CssBaseline';
import Typography from '@material-ui/core/Typography';
import Container from '@material-ui/core/Container';
import { makeStyles } from '@material-ui/core/styles';
import { shadows } from '@material-ui/system';
import { Box } from '@material-ui/core';
import TrackersList from './TrackersList'
import Map from './Map';


const useStyles = makeStyles(theme => ({
    container:{
        width: 900,
        height:500,
        marginTop:20,
        
    },
    box:{
        width:'100%',
        height:'100%',
        backgroundColor:"#fff",
    }

}));

export default function FixedContainer2() {
    const classes = useStyles();
  return (
    <React.Fragment>
      <CssBaseline />
      <Container fixed className={classes.container} >
        <Box  className={classes.box} boxShadow={2}>
            <TrackersList />
            <Map />
        </Box>
      </Container>
    </React.Fragment>
  );
}