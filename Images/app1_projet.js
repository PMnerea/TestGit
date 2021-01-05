window.addEventListener("load", function (event) {
    document.getElementByType("radio").setAttribute("checked", "false");
});

function Joueur(firstName, lastName, pseudo) {
    this.firstName = firstName;
    this.lastName = lastName;
    this.pseudo = pseudo;
    this.lcdp = "";
    this.st = "";
    this.minion = "";
    this.myst = "";
}

let joueurs = [];
let pseudos = [];

function Ajouter() {
    let pnm = document.getElementById("prenom").value;
    let nm = document.getElementById("nom").value;
    let ps = document.getElementById("psd").value;

    let joueur = new Joueur(pnm, nm, ps);

    let present = false;
    for (let i = 0; i < joueurs.length; i++) {
        if (joueur.pseudo == joueurs[i].pseudo) {
            present = true;
            break;
        }
    }
    if (joueur.pseudo.length < 1 || pnm=='' || nm=='') {
        alert('Veuiller entrer un nom d\'utilisateur valide');
    }
    else if (present) {
        alert('Ce pseudo existe déjà. Veuillez en entrer un autre.');
    }
    else {
        joueurs.push(joueur);

        const tblPers = document.getElementById("tab_j");
        const persRaw = document.createElement("TR");
        const nameCell = document.createElement("TD");
        const lcdpCell = document.createElement("TD");
        const stCell = document.createElement("TD");
        const minionCell = document.createElement("TD");
        const mystCell = document.createElement("TD");
        nameCell.innerText = joueur.pseudo;
        lcdpCell.innerText = joueur.lcdp;
        stCell.innerText = joueur.st;
        minionCell.innerText = joueur.minion;
        mystCell.innerText = joueur.myst;

        const rLcdp = (joueur.pseudo).concat('_', 'lcdp');
        const rSt = (joueur.pseudo).concat('_', 'st');
        const rMin = (joueur.pseudo).concat('_', 'mini');
        const rMyst = (joueur.pseudo).concat('_', 'myst');

        lcdpCell.setAttribute("id", rLcdp);
        stCell.setAttribute("id", rSt);
        minionCell.setAttribute("id", rMin);
        mystCell.setAttribute("id", rMyst);

        const removeCell = document.createElement("TD");
        const removeButton = document.createElement("BUTTON");

        removeButton.innerText = "Supprimer";
        removeButton.addEventListener("click", removePerson);

        removeCell.appendChild(removeButton);
        persRaw.appendChild(nameCell);
        persRaw.appendChild(lcdpCell);
        persRaw.appendChild(stCell);
        persRaw.appendChild(minionCell);
        persRaw.appendChild(mystCell);
        persRaw.appendChild(removeCell);
        tblPers.appendChild(persRaw);
    }
}


function removePerson(mouseEvent) {
    let ps = mouseEvent.target.parentElement.parentElement.firstChild.innerText;
    let index;
    for (let i = 0; i < joueurs.length; i++) {
        if (joueurs[i].pseudo == ps) {
            index = i;
            break;
        }
    }
    joueurs.splice(index, 1);
    console.log(joueurs);
    mouseEvent.target.parentElement.parentElement.remove();
}


function ResST() {
    if (joueurs.length == 0) {
        alert('Veuillez vous enregistrer dans la liste de joueurs.');
    }
    else {
        let ps = prompt('Veuillez entrer votre pseudo :');

        let present = false;
        for (let i = 0; i < joueurs.length; i++) {
            if (ps == joueurs[i].pseudo) {
                present = true;
                break;
            }
        }

        if (present) {
            let Res = { E: 0, H: 0, S: 0, D: 0 };
            try {
                var questions = [document.querySelector("input[name='st_1']:checked").value,
                document.querySelector("input[name='st_2']:checked").value,
                document.querySelector("input[name='st_3']:checked").value,
                document.querySelector("input[name='st_4']:checked").value,
                document.querySelector("input[name='st_5']:checked").value,
                document.querySelector("input[name='st_6']:checked").value,
                document.querySelector("input[name='st_7']:checked").value,
                document.querySelector("input[name='st_8']:checked").value]
            }
            catch (err) {
                alert("Toutes les réponses n'ont pas été cochées.");
            }
            for (let i = 0; i < questions.length; i++) {
                if (i == 1 || i == 6) {
                    switch (questions[i]) {
                        case 'eleven':
                            Res.E++;
                            break;
                        default:
                            Res.H++;
                            Res.S++;
                            Res.D++;
                    }
                }
                else if (i == 5) {
                    switch (questions[i]) {
                        case 'hopper':
                            Res.H++;
                            Res.S++;
                            break;
                        default:
                            Res.E++;
                            Res.D++;
                    }
                }
                else {
                    switch (questions[i]) {
                        case 'eleven':
                            Res.E++;
                            break;
                        case 'hopper':
                            Res.H++;
                            break;
                        case 'steve':
                            Res.S++;
                            break;
                        default:
                            Res.D++;
                    }
                }
            }

            let max = 0
            let pers;
            for (let i in Res) {
                if (Res[i] > max) {
                    max = Res[i];
                    pers = i;
                }
            }
            let perso;

            const hparagraphe = document.createElement('h3');
            const paragraphe = document.createElement('p');
            const image = document.createElement('img');
            const affich = document.getElementById("pSt");
            paragraphe.setAttribute("class", "resultats");
            image.setAttribute("width", 400);
            image.setAttribute("class", "ResImg");
            affich.innerHTML = '';
            if (pers == 'E') {
                perso = "Eleven";
                hparagraphe.innerText = "Eleven";
                paragraphe.innerText = "Oulala, t'as des problèmes de colère, il faut parfois prendre les évènement avec plus de recul... Mais sinon ça va, tu es une bonne personne qui donnerait tout pour ses amis et sa famille. Ps = Abuse pas trop sur les gauffres";
                image.setAttribute("src", "Images/eleven.png");
            }
            else if (pers == 'H') {
                perso = "Jim Hopper";
                hparagraphe.innerText = "Jim Hopper";
                paragraphe.innerText = "Tu es courageux et déterminé. Ton rôle de sherif te correspond parfaitement, tu cherches toujours à protéger les autres.";
                image.setAttribute("src", "Images/hopper.png");
            }
            else if (pers == 'D') {
                perso = "Dustin Henderson";
                hparagraphe.innerText = 'Dustin Henderson';
                paragraphe.innerText = "Tu es le rigolo du groupe. Toujours de bonne humeur, tu sais redonner le sourire à tes amis. Il faut juste que tu ai un peu plus confiance en toi!";
                image.setAttribute("src", "Images/dustin.png");
            }
            else {
                perso = "Steve Harrington";
                hparagraphe.innerText = 'Steve Harrington';
                paragraphe.innerText = "Tu as su grandir grâce aux aventures que tu as pu vivre. T'es le gars sympa du groupe, tes amis t'apprécient beaucoup";
                image.setAttribute("src", "Images/steve.png");
            }
            affich.appendChild(hparagraphe);
            affich.appendChild(image);
            affich.appendChild(paragraphe);

            const ID = (ps).concat('_', 'st');
            const caseRes = document.getElementById(ID);

            caseRes.innerText = perso;

        }
        else {
            alert('Ce pseudo n\'est pas valide');
        }
    }
}

function ResLCDP() {
    if (joueurs.length == 0) {
        alert('Veuillez vous enregistrer dans la liste de joueurs.');
    }
    else {
        let ps = prompt('Veuillez entrer votre pseudo :');

        let present = false;
        for (let i = 0; i < joueurs.length; i++) {
            if (ps == joueurs[i].pseudo) {
                present = true;
                break;
            }
        }

        if (present) {
            let Res = { T: 0, P: 0, H: 0, B: 0 };
            try {
                var questions = [document.querySelector("input[name='lcdp_1']:checked").value,
                document.querySelector("input[name='lcdp_2']:checked").value,
                document.querySelector("input[name='lcdp_3']:checked").value,
                document.querySelector("input[name='lcdp_4']:checked").value,
                document.querySelector("input[name='lcdp_5']:checked").value,
                document.querySelector("input[name='lcdp_6']:checked").value,
                document.querySelector("input[name='lcdp_7']:checked").value]
            }
            catch (err) {
                alert("Toutes les réponses n'ont pas été cochées.");
            }
            for (let i = 0; i < questions.length; i++) {
                switch (questions[i]) {
                    case 'professeur':
                        Res.P++;
                        break;
                    case 'tokyo':
                        Res.T++;
                        break;
                    case 'helsinki':
                        Res.H++;
                        break;
                    default:
                        Res.B++;
                }
            }

            let max = 0
            let pers;
            for (let i in Res) {
                if (Res[i] > max) {
                    max = Res[i];
                    pers = i;
                }
            }

            const hparagraphe = document.createElement('h3');
            const paragraphe = document.createElement('p');
            const image = document.createElement('img');
            const affich = document.getElementById("pLcdp");
            image.setAttribute("width", 400);
            image.setAttribute("class", "ResImg");
            paragraphe.setAttribute("class", "resultats");
            affich.innerHTML = '';

            if (pers == 'T') {
                perso = "Tokyo";
                hparagraphe.innerText = "Tokyo";
                paragraphe.innerText = "Tu es fort, intrépide et sûr de toi mais quand on s'en prend à une personne que tu aimes, tu ne te contrôle plus et tu fais tout pour les venger, tu es aussi très humble et gentil malgré tes allures de rebelle.";
                image.setAttribute("src", "Images/tokyo.png");
            }
            else if (pers == 'P') {
                perso = "Le professeur";
                hparagraphe.innerText = "Le professeur";
                paragraphe.innerText = "Tu réfléchis avant d'agir, prends ton temps et tu n'es pas brusque, ce que tu entreprends est sûr d'être réussi, tu serais un parfait professeur!";
                image.setAttribute("src", "Images/profesor.png");
            }
            else if (pers == 'H') {
                perso = "Helsinki";
                hparagraphe.innerText = "Helsinki";
                paragraphe.innerText = 'Pour toi un plan doit être violent mais pas que! Tu as aussi un grand coeur et tu n\'hésite pas à aider tes proches quand ils en ont besoin';
                image.setAttribute("src", "Images/helsinki.png");
            }
            else {
                perso = "Berlin";
                hparagraphe.innerText = "Berlin";
                paragraphe.innerText = "Tu aimes le pouvoir et tu n'as peur de rien, tu est souvent pessimiste comme Berlin, tu es casanier et très solidaire, personne ne peut t'empêcher de terminer ce que tu as commencé!";
                image.setAttribute("src", "Images/berlin.png");
            }
            affich.appendChild(hparagraphe);
            affich.appendChild(image);
            affich.appendChild(paragraphe);

            const ID = (ps).concat('_', 'lcdp');
            const caseRes = document.getElementById(ID);

            caseRes.innerText = perso;
        }
        else {
            alert('Ce pseudo n\'est pas valide');
        }
    }
}

function ResMini() {
    if (joueurs.length == 0) {
        alert("Veuillez vous enregistrer dans la liste de joueurs.");
    }
    else {
        let ps = prompt('Veuillez entrer votre pseudo :');

        let present = false;
        for (let i = 0; i < joueurs.length; i++) {
            if (ps == joueurs[i].pseudo) {
                present = true;
                break;
            }
        }

        if (present) {

            let Res = { B: 0, S: 0, K: 0 };
            try {
                var questions = [document.querySelector("input[name='min_1']:checked").value,
                document.querySelector("input[name='min_2']:checked").value,
                document.querySelector("input[name='min_3']:checked").value,
                document.querySelector("input[name='min_4']:checked").value,
                document.querySelector("input[name='min_5']:checked").value,
                document.querySelector("input[name='min_6']:checked").value]
            }
            catch (err) {
                alert("Toutes les réponses n'ont pas été cochées.");
            }
            for (let i = 0; i < questions.length; i++) {
                switch (questions[i]) {
                    case 'bob':
                        Res.B++;
                        break;
                    case 'stuart':
                        Res.S++;
                        break;
                    default:
                        Res.K++;
                }
            }

            let max = 0
            let pers;
            for (let i in Res) {
                if (Res[i] > max) {
                    max = Res[i];
                    pers = i;
                }
            }

            const hparagraphe = document.createElement('h3');
            const paragraphe = document.createElement('p');
            const image = document.createElement('img');
            paragraphe.setAttribute("class", "resultats");
            const affich = document.getElementById("pMini");
            image.setAttribute("width", 400);
            image.setAttribute("class", "ResImg");
            affich.innerHTML = '';

            if (pers == 'B') {
                perso = "Bob";
                hparagraphe.innerText = "Bob";
                paragraphe.innerText = "Tu aimes t'occuper de la tribu et de ton doudou. Des \"bananas\" aussi! Tu es bien mignon mais aussi bien plus démoniaque quand tu le veux!";
                image.setAttribute("src", "Images/bob.png");
            }
            else if (pers == 'S') {
                perso = "Stuart";
                hparagraphe.innerText = "Stuart";
                paragraphe.innerText = "Tu aimes beaucoup la musique, en particulier la guitare. Mais ce que tu préfères, c'est la \"banana\"! Bien sûr, si on a besoin de toi, tu pourras toujours répondre \"présent\"!";
                image.setAttribute("src", "Images/stuart.png");
            }
            else {
                perso = "Kevin";
                hparagraphe.innerText = "Kevin";
                paragraphe.innerText = "Petit intello, va! Mais aussi grand rigolo! Sans toi, la saga \"Minions\" n'existerait pas sans doute. Je me trompe? Ah j'oubliais! Tu es officiellement fan des \"bananas\".";
                image.setAttribute("src", "Images/kevin.png");
            }

            affich.appendChild(hparagraphe);
            affich.appendChild(image);
            affich.appendChild(paragraphe);

            const ID = (ps).concat('_', 'mini');
            const caseRes = document.getElementById(ID);

            caseRes.innerText = perso;
        }
        else {
            alert('Ce pseudo n\'est pas valide');
        }
    }

}

function ResMyst() {
    if (joueurs.length == 0) {
        alert("Veuillez vous enregistrer dans la liste de joueurs.");
    }
    else {
        let ps = prompt('Veuillez entrer votre pseudo :');

        let present = false;
        for (let i = 0; i < joueurs.length; i++) {
            if (ps == joueurs[i].pseudo) {
                present = true;
                break;
            }
        }

        if (present) {

            let Res = { Ca: 0, Cl: 0, N: 0, Ch: 0 };
            try {
                var questions = [document.querySelector("input[name='myst_1']:checked").value,
                document.querySelector("input[name='myst_2']:checked").value,
                document.querySelector("input[name='myst_3']:checked").value,
                document.querySelector("input[name='myst_4']:checked").value]
            }
            catch (err) {
                alert("Toutes les réponses n'ont pas été cochées.");
            }
            for (let i = 0; i < questions.length; i++) {
                if (i == 3) { 
                    switch (questions[i]) {
                        case 'carina':
                            Res.Ca++;
                            break;
                        case 'clem':
                            Res.Cl++;
                            Res.N++;
                            break;
                        default:
                            Res.Ch++;
                    }
                }
                else {
                    switch (questions[i]) {
                        case 'carina':
                            Res.Ca++;
                            break;
                        case 'clem':
                            Res.Cl++;
                            break;
                        case 'nerea':
                            Res.N++;
                            break;
                        default:
                            Res.Ch++;
                    }
                }
            }

            let max = 0
            let pers;
            for (let i in Res) {
                if (Res[i] > max) {
                    max = Res[i];
                    pers = i;
                }
            }

            const hparagraphe = document.createElement('h3');
            const paragraphe = document.createElement('p');
            const image = document.createElement('img');
            paragraphe.setAttribute("class", "resultats");
            const affich = document.getElementById("pMyst");
            image.setAttribute("width", 400);
            image.setAttribute("class", "ResImg");
            affich.innerHTML = '';

            if (pers == 'Ca') {
                perso = "Carina";
                hparagraphe.innerText = "Carina";
                paragraphe.innerText = "Bravo! comme elle tu as gagné, ta récompense est de savoir que tu ressemble à Carina.";
                image.setAttribute("src", "Images/carina.png");
            }
            else if (pers == 'Cl') {
                perso = "Clémence";
                hparagraphe.innerText = "Clémence";
                paragraphe.innerText = "On a qu'une seule vie après tout. Félicitation! Tu es une deuxième Clémence.";
                image.setAttribute("src", "Images/clem.jpg");
            }
            else if (pers == 'N') {
                perso = "Nerea";
                hparagraphe.innerText = "Nerea";
                paragraphe.innerText = "Excellent, tu es une deuxième Nerea. Tu peux être aussi heureux qu'elle quand elle a trouvé ses initiales.";
                image.setAttribute("src", "Images/nerea.jpg");
            }
            else {
                perso = "Chama";
                hparagraphe.innerText = "Chama";
                paragraphe.innerText = "Excellent, tu ressemble beaucoup à Chama";
                image.setAttribute("src", "Images/chama.jpg");
            }

            affich.appendChild(hparagraphe);
            affich.appendChild(image);
            affich.appendChild(paragraphe);

            const ID = (ps).concat('_', 'myst');
            const caseRes = document.getElementById(ID);

            caseRes.innerText = perso;
        }
        else {
            alert('Ce pseudo n\'est pas valide');
        }
    }

}
